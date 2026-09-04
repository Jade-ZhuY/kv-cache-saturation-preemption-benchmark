#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "httpx>=0.27,<1",
#   "matplotlib>=3.8,<4",
#   "transformers>=4.40,<5",
# ]
# ///
"""Closed-loop KV-cache capacity benchmark for OpenAI-compatible vLLM APIs.

Run directly with: ``uv run kv_cache_benchmark.py --model MODEL ...``.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import re
import shutil
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import httpx
except ImportError as exc:  # pragma: no cover - exercised by an operator
    raise SystemExit("缺少依赖，请先执行: python -m pip install -r requirements.txt") from exc

DEFAULT_CONCURRENCIES = "1,2,4,8,16,32"
SSE_PREFIX = "data:"

# vLLM has renamed several metrics across releases.  The collector recognises
# both common names and saves the matching original metric names for auditing.
METRIC_ALIASES = {
    "kv_cache_usage": (
        "vllm:gpu_cache_usage_perc",
        "vllm:kv_cache_usage_perc",
        "vllm:gpu_kv_cache_usage_perc",
    ),
    "running_requests": ("vllm:num_requests_running",),
    "waiting_requests": ("vllm:num_requests_waiting",),
    "preemptions_total": ("vllm:num_preemptions_total", "vllm:num_preemptions"),
    "queue_time_sum": ("vllm:request_queue_time_seconds_sum",),
    "queue_time_count": ("vllm:request_queue_time_seconds_count",),
}
COUNTER_LOGICAL_NAMES = {"preemptions_total", "queue_time_sum", "queue_time_count"}


@dataclass
class RequestResult:
    request_id: int
    started_s: float
    finished_s: float | None
    ttft_s: float | None
    e2e_s: float | None
    tpot_s: float | None
    prompt_tokens: int | None
    output_tokens: int | None
    status: str
    error: str | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(values: Iterable[float], percentile_value: float) -> float | None:
    """Linear-interpolated percentile; returns None for an empty sample."""
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * percentile_value / 100.0
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def rounded(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(value, digits)


def terminal_metric(value: Any) -> str:
    """Keep the per-concurrency progress line readable when a metric is absent."""
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "N/A"
    return str(int(value)) if float(value).is_integer() else f"{float(value):.2f}"


def terminal_percent(value: Any) -> str:
    """Format percentage metrics for the end-of-run comparison table."""
    rendered = terminal_metric(value)
    return rendered if rendered == "N/A" else f"{rendered}%"


def render_concurrency_comparison(summaries: list[dict[str, Any]]) -> None:
    """Print the small comparison table that complements the KV utilization plot."""
    headers = ("Concurrency", "Success", "KV Avg", "KV Peak", "Waiting Peak", "Preemptions")
    rows: list[tuple[str, ...]] = []
    for summary in summaries:
        metrics = summary["server_metrics"]
        counts = summary["request_counts"]
        rows.append((
            str(summary["configuration"]["concurrency"]),
            f"{counts['successful']}/{counts['attempted']}",
            terminal_percent(metrics.get("kv_cache_usage_avg")),
            terminal_percent(metrics.get("kv_cache_usage_peak")),
            terminal_metric(metrics.get("waiting_requests_peak")),
            terminal_metric(metrics.get("preemptions_total_delta")),
        ))
    widths = [max(len(header), *(len(row[index]) for row in rows)) for index, header in enumerate(headers)]
    separator = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    print("\n并发对比：")
    print(separator)
    print("|" + "|".join(f" {header:<{widths[index]}} " for index, header in enumerate(headers)) + "|")
    print(separator)
    for row in rows:
        print("|" + "|".join(f" {value:<{widths[index]}} " for index, value in enumerate(row)) + "|")
    print(separator, flush=True)


def numeric_series(summaries: list[dict[str, Any]], section: str, key: str) -> list[float]:
    """Return a plotting series; unavailable metrics become gaps instead of zero."""
    values: list[float] = []
    for summary in summaries:
        value = summary.get(section, {}).get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
        else:
            values.append(float("nan"))
    return values


def capacity_assessment(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """State the observed cache threshold without conflating it with compute saturation."""
    concurrencies = [summary["configuration"]["concurrency"] for summary in summaries]
    peaks = numeric_series(summaries, "server_metrics", "kv_cache_usage_peak")
    valid = [(concurrency, peak) for concurrency, peak in zip(concurrencies, peaks) if not math.isnan(peak)]
    if not valid:
        return {
            "status": "no_kv_cache_metric",
            "message": "未采集到 KV Cache 指标；请检查 --metrics-url 是否指向 vLLM /metrics。",
        }
    saturated = next(((concurrency, peak) for concurrency, peak in valid if peak >= 95.0), None)
    max_concurrency, max_peak = max(valid, key=lambda item: item[1])
    if saturated:
        return {
            "status": "kv_cache_saturated",
            "saturation_concurrency": saturated[0],
            "saturation_peak_percent": rounded(saturated[1], 2),
            "message": f"KV Cache 在 concurrency={saturated[0]} 首次达到 {saturated[1]:.2f}%（≥95% 饱和阈值）。",
        }
    if max_peak >= 85.0:
        return {
            "status": "kv_cache_pressure",
            "pressure_concurrency": max_concurrency,
            "peak_percent": rounded(max_peak, 2),
            "message": f"KV Cache 最高为 {max_peak:.2f}%（≥85% 压力区，但尚未达到 95% 饱和阈值）。",
        }
    return {
        "status": "kv_cache_not_saturated",
        "peak_concurrency": max_concurrency,
        "peak_percent": rounded(max_peak, 2),
        "message": f"最高并发 concurrency={max_concurrency} 时 KV Cache 峰值为 {max_peak:.2f}%，尚未进入压力区。",
    }


def generate_capacity_plot(summaries: list[dict[str, Any]], output_dir: Path) -> tuple[Path | None, dict[str, Any]]:
    """Generate the single KV Cache utilization curve used for capacity decisions."""
    assessment = capacity_assessment(summaries)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        assessment["plot_error"] = "matplotlib 未安装，无法生成曲线图。"
        return None, assessment

    concurrencies = [summary["configuration"]["concurrency"] for summary in summaries]
    x = list(range(len(concurrencies)))
    labels = [str(value) for value in concurrencies]
    kv_average = numeric_series(summaries, "server_metrics", "kv_cache_usage_avg")
    kv_peak = numeric_series(summaries, "server_metrics", "kv_cache_usage_peak")
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 11,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(8.4, 6.2), constrained_layout=True)

    # The two highlighted bands mirror the experiment plan: 85–95% is the
    # pressure range; usage at or above 95% is the saturation range.
    ax.axhspan(85, 95, color="#f4a261", alpha=0.18, label="Pressure zone (85-95%)")
    ax.axhspan(95, 100, color="#e76f51", alpha=0.22, label="Saturation zone (>=95%)")
    ax.axhline(85, color="#e9c46a", linewidth=1, linestyle="--")
    ax.axhline(95, color="#e76f51", linewidth=1, linestyle="--")
    if not all(math.isnan(value) for value in kv_average):
        ax.plot(x, kv_average, "o-", linewidth=2, color="#457b9d", label="KV avg")
    if not all(math.isnan(value) for value in kv_peak):
        ax.plot(x, kv_peak, "o-", linewidth=2.5, color="#d62828", label="KV peak")
    if assessment.get("status") == "kv_cache_saturated":
        index = concurrencies.index(assessment["saturation_concurrency"])
        ax.axvline(index, color="#d62828", linewidth=1.5, linestyle=":")
        ax.annotate(
            f"Saturation\nC={assessment['saturation_concurrency']}",
            xy=(index, kv_peak[index]), xytext=(8, -34), textcoords="offset points",
            arrowprops={"arrowstyle": "->", "color": "#d62828"}, color="#d62828", fontsize=10,
        )
    ax.set_title("KV Cache utilization")
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Usage (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(x, labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    if all(math.isnan(value) for value in kv_peak):
        ax.text(0.5, 0.48, "No KV Cache metric\nSet --metrics-url", transform=ax.transAxes, ha="center", va="center")

    plot_path = output_dir / "kv_cache_utilization.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return plot_path, assessment


def parse_concurrencies(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--concurrency 必须是正整数或逗号分隔的正整数") from exc
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("--concurrency 必须包含至少一个正整数")
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("--concurrency 中不能有重复值")
    return result


class TokenCounter:
    """Uses the deployment tokenizer when supplied, otherwise labels estimates."""

    def __init__(self, tokenizer_name: str | None, chars_per_token: float) -> None:
        self._tokenizer: Any | None = None
        self.chars_per_token = chars_per_token
        self.mode = "estimated_chars"
        if tokenizer_name:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise SystemExit(
                    "--tokenizer 需要 transformers，请先执行: python -m pip install -r requirements.txt"
                ) from exc
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            self.mode = f"transformers:{tokenizer_name}"

    @property
    def exact(self) -> bool:
        return self._tokenizer is not None

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text, add_special_tokens=False))
        return max(1, round(len(text) / self.chars_per_token))

    def make_prompt(self, requested_tokens: int, nonce: str | None = None) -> tuple[str, int]:
        # Repeated technical prose avoids an unrealistically cheap highly
        # repetitive token sequence while remaining deterministic between runs.
        seed = (
            "KV cache capacity benchmark request. Explain the relationship between "
            "attention state, batching, scheduling latency, and token throughput. "
        )
        # Put a per-request nonce at the beginning.  Prefix-cache block hashes
        # are parent-dependent in vLLM, so changing the initial block prevents
        # a repeated synthetic prompt from silently becoming a prefix-cache
        # benchmark instead of a KV-capacity benchmark.
        prefix = f"Benchmark request nonce={nonce}. " if nonce is not None else ""
        if self._tokenizer is not None:
            prefix_ids = self._tokenizer.encode(prefix, add_special_tokens=False)
            seed_ids = self._tokenizer.encode(seed, add_special_tokens=False)
            remaining = max(0, requested_tokens - len(prefix_ids))
            repetitions = math.ceil(remaining / max(1, len(seed_ids))) + 1
            ids = (prefix_ids + seed_ids * repetitions)[:requested_tokens]
            text = self._tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            # Decoding may merge boundary whitespace for a few tokenizer types.
            # Trim once more to guarantee that the local content is never longer.
            ids = self._tokenizer.encode(text, add_special_tokens=False)[:requested_tokens]
            text = self._tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            return text, len(ids)
        target_chars = max(1, round(requested_tokens * self.chars_per_token))
        text = (prefix + seed * math.ceil(target_chars / len(seed)))[:target_chars]
        return text, self.count(text)


def parse_prometheus(payload: str) -> dict[str, list[float]]:
    """Parse Prometheus exposition text and group values by base metric name."""
    result: dict[str, list[float]] = defaultdict(list)
    pattern = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{[^}]*\})?\s+([-+0-9.eE]+)(?:\s+\d+)?$")
    for line in payload.splitlines():
        if not line or line.startswith("#"):
            continue
        match = pattern.match(line.strip())
        if not match:
            continue
        try:
            value = float(match.group(2))
        except ValueError:
            continue
        if math.isfinite(value):
            result[match.group(1)].append(value)
    return result


def metric_value(metric_map: dict[str, list[float]], logical_name: str) -> tuple[float | None, list[str]]:
    """Aggregate a logical vLLM metric and return the raw metric names used."""
    aliases = METRIC_ALIASES[logical_name]
    matches = [(name, values) for name, values in metric_map.items() if name in aliases]
    # Fallback handles minor metric renames without treating unrelated cache
    # counters (for example prefix-cache hits) as capacity usage.
    if not matches:
        if logical_name == "kv_cache_usage":
            matches = [
                (name, values)
                for name, values in metric_map.items()
                if "cache_usage" in name and ("gpu" in name or "kv" in name)
            ]
        elif logical_name == "preemptions_total":
            matches = [(name, values) for name, values in metric_map.items() if "preemption" in name]
        elif logical_name in {"running_requests", "waiting_requests"}:
            needle = "running" if logical_name.startswith("running") else "waiting"
            matches = [
                (name, values)
                for name, values in metric_map.items()
                if "request" in name and needle in name and "num" in name
            ]
    if not matches:
        return None, []
    flattened = [value for _, values in matches for value in values]
    # Multiple labelled engine metrics can exist. A nearly full single cache is
    # important, whereas requests/counters should be summed across engines.
    if logical_name == "kv_cache_usage":
        return max(flattened), [name for name, _ in matches]
    return sum(flattened), [name for name, _ in matches]


async def query_nvidia_smi() -> dict[str, float] | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        process = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=5)
    except (OSError, asyncio.TimeoutError):
        return None
    if process.returncode != 0:
        return None
    rows: list[tuple[float, float, float, float | None]] = []
    for line in stdout.decode("utf-8", errors="replace").splitlines():
        try:
            fields = [
                None if field.strip().upper().strip("[]") == "N/A" else float(field.strip())
                for field in line.split(",")
            ]
        except ValueError:
            continue
        if len(fields) == 4 and all(value is not None and math.isfinite(value) for value in fields[:3]):
            power = fields[3] if fields[3] is not None and math.isfinite(fields[3]) else None
            rows.append((fields[0], fields[1], fields[2], power))
    if not rows:
        return None
    result = {
        "gpu_utilization_pct": statistics.fmean(row[0] for row in rows),
        "gpu_memory_used_mib": sum(row[1] for row in rows),
        "gpu_memory_total_mib": sum(row[2] for row in rows),
    }
    powers = [row[3] for row in rows if row[3] is not None]
    if powers:
        result["gpu_power_w"] = sum(powers)
    return result


class MetricsSampler:
    def __init__(
        self,
        client: httpx.AsyncClient,
        metrics_url: str | None,
        interval_s: float,
        include_gpu_smi: bool,
        origin: float,
    ) -> None:
        self.client = client
        self.metrics_url = metrics_url
        self.interval_s = interval_s
        self.include_gpu_smi = include_gpu_smi
        self.origin = origin
        self.samples: list[dict[str, Any]] = []
        self.source_names: set[str] = set()
        self.errors: list[str] = []
        self._stop = asyncio.Event()

    async def run(self) -> None:
        # Take an initial sample so counter deltas include the whole test.
        while True:
            await self._take_sample()
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval_s)
                # Counters need an end sample; otherwise a short run can report
                # a zero preemption/queue delta even though events occurred.
                await self._take_sample()
                break
            except asyncio.TimeoutError:
                pass

    def stop(self) -> None:
        self._stop.set()

    async def _take_sample(self) -> None:
        sample: dict[str, Any] = {"elapsed_s": time.perf_counter() - self.origin}
        if self.metrics_url:
            try:
                response = await self.client.get(self.metrics_url)
                response.raise_for_status()
                metric_map = parse_prometheus(response.text)
                for logical_name in METRIC_ALIASES:
                    value, names = metric_value(metric_map, logical_name)
                    if value is not None:
                        sample[logical_name] = value
                        self.source_names.update(names)
            except (httpx.HTTPError, ValueError) as exc:
                self.errors.append(f"metrics: {type(exc).__name__}: {exc}")
        if self.include_gpu_smi:
            gpu = await query_nvidia_smi()
            if gpu:
                sample.update(gpu)
        if len(sample) > 1:
            self.samples.append(sample)

    def summarize(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "samples": len(self.samples),
            "source_metric_names": sorted(self.source_names),
            "errors": self.errors[:10],
        }
        for logical_name in ("kv_cache_usage", "running_requests", "waiting_requests"):
            values = [float(sample[logical_name]) for sample in self.samples if logical_name in sample]
            if values:
                average = statistics.fmean(values)
                peak = max(values)
                # vLLM's *_usage_perc gauges are normally ratios (0–1). Store
                # capacity usage in percent (0–100) in the summaries/CSV while
                # retaining the untouched Prometheus values in samples JSONL.
                if logical_name == "kv_cache_usage" and peak <= 1.0:
                    average, peak = average * 100.0, peak * 100.0
                result[f"{logical_name}_avg"] = average
                result[f"{logical_name}_peak"] = peak
        for logical_name in ("gpu_utilization_pct", "gpu_memory_used_mib", "gpu_power_w"):
            values = [float(sample[logical_name]) for sample in self.samples if logical_name in sample]
            if values:
                result[f"{logical_name}_avg"] = statistics.fmean(values)
                result[f"{logical_name}_peak"] = max(values)
        for logical_name in COUNTER_LOGICAL_NAMES:
            values = [float(sample[logical_name]) for sample in self.samples if logical_name in sample]
            if len(values) >= 2:
                result[f"{logical_name}_delta"] = max(0.0, values[-1] - values[0])
        queue_sum = result.get("queue_time_sum_delta")
        queue_count = result.get("queue_time_count_delta")
        if queue_sum is not None and queue_count and queue_count > 0:
            result["queue_time_mean_s"] = queue_sum / queue_count
        return result


def build_payload(args: argparse.Namespace, prompt: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "max_tokens": args.output_tokens,
        "temperature": args.temperature,
        "stream": True,
    }
    if args.ignore_eos:
        payload["ignore_eos"] = True
    if args.api_mode == "chat":
        payload["messages"] = [{"role": "user", "content": prompt}]
    else:
        payload["prompt"] = prompt
    if not args.no_stream_usage:
        payload["stream_options"] = {"include_usage": True}
    return payload


def response_text(choice: dict[str, Any], api_mode: str) -> str:
    if api_mode == "chat":
        delta = choice.get("delta") or {}
        content = delta.get("content", "")
        if isinstance(content, list):  # Some OpenAI-compatible multimodal APIs.
            return "".join(item.get("text", "") for item in content if isinstance(item, dict))
        return content if isinstance(content, str) else ""
    text = choice.get("text", "")
    return text if isinstance(text, str) else ""


async def execute_request(
    request_id: int,
    client: httpx.AsyncClient,
    endpoint: str,
    payload: dict[str, Any],
    args: argparse.Namespace,
    token_counter: TokenCounter,
    prompt_tokens_fallback: int,
    origin: float,
) -> RequestResult:
    started_absolute = time.perf_counter()
    first_token_absolute: float | None = None
    generated_parts: list[str] = []
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    try:
        async with client.stream("POST", endpoint, json=payload) as response:
            response.raise_for_status()
            async for raw_line in response.aiter_lines():
                line = raw_line.strip()
                if not line or not line.startswith(SSE_PREFIX):
                    continue
                data = line[len(SSE_PREFIX) :].strip()
                if data == "[DONE]":
                    continue
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    # Keep consuming: some proxies inject non-JSON SSE events.
                    continue
                usage = event.get("usage")
                if isinstance(usage, dict):
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    output_tokens = usage.get("completion_tokens", output_tokens)
                choices = event.get("choices") or []
                if choices and isinstance(choices[0], dict):
                    text = response_text(choices[0], args.api_mode)
                    if text:
                        if first_token_absolute is None:
                            first_token_absolute = time.perf_counter()
                        generated_parts.append(text)
        finished_absolute = time.perf_counter()
        if prompt_tokens is None:
            prompt_tokens = prompt_tokens_fallback
        if output_tokens is None:
            output_tokens = token_counter.count("".join(generated_parts))
        e2e = finished_absolute - started_absolute
        ttft = (first_token_absolute - started_absolute) if first_token_absolute else None
        tpot = None
        if ttft is not None and output_tokens and output_tokens > 1:
            tpot = max(0.0, (finished_absolute - first_token_absolute) / (output_tokens - 1))
        return RequestResult(
            request_id=request_id,
            started_s=started_absolute - origin,
            finished_s=finished_absolute - origin,
            ttft_s=ttft,
            e2e_s=e2e,
            tpot_s=tpot,
            prompt_tokens=int(prompt_tokens) if prompt_tokens is not None else None,
            output_tokens=int(output_tokens) if output_tokens is not None else None,
            status="ok",
        )
    except (httpx.HTTPError, asyncio.TimeoutError, OSError, ValueError) as exc:
        finished_absolute = time.perf_counter()
        return RequestResult(
            request_id=request_id,
            started_s=started_absolute - origin,
            finished_s=finished_absolute - origin,
            ttft_s=None,
            e2e_s=finished_absolute - started_absolute,
            tpot_s=None,
            prompt_tokens=None,
            output_tokens=None,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
        )


async def run_once(args: argparse.Namespace, concurrency: int, run_dir: Path) -> dict[str, Any]:
    token_counter = TokenCounter(args.tokenizer, args.chars_per_token)
    _, generated_prompt_tokens = token_counter.make_prompt(args.input_tokens, nonce="shape-check")
    endpoint = args.base_url.rstrip("/") + ("/chat/completions" if args.api_mode == "chat" else "/completions")
    origin = time.perf_counter()
    headers = {"Accept": "text/event-stream"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    timeout = httpx.Timeout(args.request_timeout_s, connect=args.connect_timeout_s)
    limits = httpx.Limits(max_connections=max(concurrency + 4, 10), max_keepalive_connections=max(concurrency, 1))
    results: list[RequestResult] = []
    queue: asyncio.Queue[int] = asyncio.Queue()
    for request_id in range(args.requests):
        queue.put_nowait(request_id)

    async with httpx.AsyncClient(headers=headers, timeout=timeout, limits=limits) as client:
        sampler = MetricsSampler(
            client=client,
            metrics_url=args.metrics_url,
            interval_s=args.metrics_interval_s,
            include_gpu_smi=args.gpu_smi,
            origin=origin,
        )
        sampler_task = asyncio.create_task(sampler.run())

        async def worker() -> None:
            while True:
                try:
                    request_id = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                prompt, prompt_tokens_fallback = token_counter.make_prompt(
                    args.input_tokens,
                    nonce=None if args.reuse_prompt else f"c{concurrency}-r{request_id}",
                )
                result = await execute_request(
                    request_id,
                    client,
                    endpoint,
                    build_payload(args, prompt),
                    args,
                    token_counter,
                    prompt_tokens_fallback,
                    origin,
                )
                results.append(result)
                queue.task_done()

        await asyncio.gather(*(worker() for _ in range(min(concurrency, args.requests))))
        sampler.stop()
        await sampler_task

    ended = time.perf_counter()
    result_dicts = [asdict(result) for result in sorted(results, key=lambda item: item.request_id)]
    write_jsonl(run_dir / "requests.jsonl", result_dicts)
    write_jsonl(run_dir / "metrics_samples.jsonl", sampler.samples)
    metrics = sampler.summarize()
    summary = summarize_run(
        args=args,
        concurrency=concurrency,
        origin=origin,
        end=ended,
        results=results,
        metrics=metrics,
        token_counter=token_counter,
        generated_prompt_tokens=generated_prompt_tokens,
        endpoint=endpoint,
    )
    write_json(run_dir / "summary.json", summary)
    return summary


def summarize_run(
    args: argparse.Namespace,
    concurrency: int,
    origin: float,
    end: float,
    results: list[RequestResult],
    metrics: dict[str, Any],
    token_counter: TokenCounter,
    generated_prompt_tokens: int,
    endpoint: str,
) -> dict[str, Any]:
    successful = [item for item in results if item.status == "ok"]
    elapsed = end - origin
    prompt_total = sum(item.prompt_tokens or 0 for item in successful)
    output_total = sum(item.output_tokens or 0 for item in successful)
    ttft = [item.ttft_s for item in successful if item.ttft_s is not None]
    tpot = [item.tpot_s for item in successful if item.tpot_s is not None]
    e2e = [item.e2e_s for item in successful if item.e2e_s is not None]

    def latency_summary(prefix: str, values: list[float]) -> dict[str, float | None]:
        return {
            f"{prefix}_p50_s": rounded(percentile(values, 50)),
            f"{prefix}_p95_s": rounded(percentile(values, 95)),
            f"{prefix}_p99_s": rounded(percentile(values, 99)),
            f"{prefix}_mean_s": rounded(statistics.fmean(values) if values else None),
        }

    summary: dict[str, Any] = {
        "schema_version": 1,
        "timestamp_utc": utc_now(),
        "configuration": {
            "base_url": args.base_url,
            "endpoint": endpoint,
            "model": args.model,
            "api_mode": args.api_mode,
            "concurrency": concurrency,
            "requests": args.requests,
            "requested_input_tokens": args.input_tokens,
            "generated_prompt_tokens": generated_prompt_tokens,
            "requested_output_tokens": args.output_tokens,
            "temperature": args.temperature,
            "ignore_eos": args.ignore_eos,
            "unique_prompts": not args.reuse_prompt,
            "token_counting": token_counter.mode,
            "metrics_url": args.metrics_url,
            "gpu_smi": args.gpu_smi,
        },
        "request_counts": {
            "attempted": len(results),
            "successful": len(successful),
            "failed": len(results) - len(successful),
        },
        "elapsed_s": rounded(elapsed),
        "throughput": {
            "request_per_s": rounded(len(successful) / elapsed if elapsed else None),
            "input_tokens_per_s": rounded(prompt_total / elapsed if elapsed else None),
            "output_tokens_per_s": rounded(output_total / elapsed if elapsed else None),
            "total_tokens_per_s": rounded((prompt_total + output_total) / elapsed if elapsed else None),
            "input_tokens_total": prompt_total,
            "output_tokens_total": output_total,
        },
        "latency": {
            **latency_summary("ttft", ttft),
            **latency_summary("tpot", tpot),
            **latency_summary("e2e", e2e),
        },
        "server_metrics": {key: rounded(value) if isinstance(value, float) else value for key, value in metrics.items()},
    }
    return summary


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    config = summary["configuration"]
    counts = summary["request_counts"]
    row.update({
        "concurrency": config["concurrency"],
        "attempted": counts["attempted"],
        "successful": counts["successful"],
        "failed": counts["failed"],
        "elapsed_s": summary["elapsed_s"],
    })
    row.update(summary["throughput"])
    row.update(summary["latency"])
    row.update(summary["server_metrics"])
    return row


def write_matrix_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    rows = [flatten_summary(summary) for summary in summaries]
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KV Cache 容量压力测试（OpenAI 兼容 vLLM API，闭环固定并发）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="OpenAI 兼容 API 的 /v1 地址")
    parser.add_argument("--model", required=True, help="vLLM 暴露的模型名")
    parser.add_argument(
        "--concurrency",
        type=parse_concurrencies,
        default=parse_concurrencies(DEFAULT_CONCURRENCIES),
        help="目标并发数；单值如 16，或矩阵如 1,2,4,8,16,32",
    )
    parser.add_argument("--requests", type=int, default=200, help="每一个并发档的总请求数")
    parser.add_argument("--input-tokens", type=int, default=8192, help="每个请求的目标输入 token 数")
    parser.add_argument("--output-tokens", type=int, default=512, help="每个请求的最大生成 token 数")
    parser.add_argument("--api-mode", choices=("completion", "chat"), default="completion", help="使用 /completions 或 /chat/completions")
    parser.add_argument("--tokenizer", help="模型的 Hugging Face tokenizer 路径或仓库；用于精确构造和计数 token")
    parser.add_argument("--chars-per-token", type=float, default=4.0, help="不传 --tokenizer 时的 token 估算比例")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--ignore-eos", action="store_true", help="要求 vLLM 忽略 EOS，尽量生成到 --output-tokens")
    parser.add_argument(
        "--reuse-prompt",
        action="store_true",
        help="所有请求使用相同 prompt；仅在有意测试 Prefix Cache 时使用",
    )
    parser.add_argument("--no-stream-usage", action="store_true", help="不发送 stream_options.include_usage（旧服务兼容模式）")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"), help="API Key；默认读取 OPENAI_API_KEY")
    parser.add_argument("--metrics-url", help="vLLM Prometheus /metrics 地址；传入后采集 KV/调度指标")
    parser.add_argument("--metrics-interval-s", type=float, default=1.0, help="Prometheus 和 GPU 采样间隔")
    parser.add_argument("--gpu-smi", action="store_true", help="本机可执行 nvidia-smi 时采集 GPU 利用率、显存和功耗")
    parser.add_argument("--connect-timeout-s", type=float, default=20.0)
    parser.add_argument("--request-timeout-s", type=float, default=3600.0)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.requests <= 0 or args.input_tokens <= 0 or args.output_tokens <= 0:
        parser.error("--requests、--input-tokens、--output-tokens 必须大于 0")
    if args.metrics_interval_s <= 0 or args.chars_per_token <= 0:
        parser.error("采样间隔和 --chars-per-token 必须大于 0")
    return args


async def async_main(args: argparse.Namespace) -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = args.output_dir / timestamp
    root.mkdir(parents=True, exist_ok=False)
    write_json(
        root / "command.json",
        {
            "argv": sys.argv,
            "timestamp_utc": utc_now(),
            "note": "每个并发档独立执行，requests.jsonl 保留逐请求延迟，metrics_samples.jsonl 保留采样值。",
        },
    )
    summaries: list[dict[str, Any]] = []
    for concurrency in args.concurrency:
        run_dir = root / f"concurrency_{concurrency}"
        run_dir.mkdir()
        print(f"[开始] concurrency={concurrency}, requests={args.requests}", flush=True)
        summary = await run_once(args, concurrency, run_dir)
        summaries.append(summary)
        metrics = summary["server_metrics"]
        print(
            f"[完成] concurrency={concurrency}; ok={summary['request_counts']['successful']}/"
            f"{summary['request_counts']['attempted']}; "
            f"waiting_peak={terminal_metric(metrics.get('waiting_requests_peak'))}; "
            f"preemptions={terminal_metric(metrics.get('preemptions_total_delta'))}",
            flush=True,
        )
    write_json(root / "matrix_summary.json", summaries)
    write_matrix_csv(root / "matrix_summary.csv", summaries)
    plot_path, _assessment = generate_capacity_plot(summaries, root)
    render_concurrency_comparison(summaries)
    if plot_path:
        print(f"容量曲线图: {plot_path.resolve()}", flush=True)
    print(f"结果目录: {root.resolve()}", flush=True)
    return 0 if all(item["request_counts"]["failed"] == 0 for item in summaries) else 2


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("已中断。已完成档位的结果保存在输出目录中。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
