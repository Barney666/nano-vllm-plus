import argparse
import json
import os
import subprocess
import sys
import time
from statistics import mean
from time import perf_counter

from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams
from example import build_long_context_prompt


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = int((len(values) - 1) * p)
    return values[k]


def build_arrival_events(tokenizer):
    long_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": build_long_context_prompt()}],
        tokenize=False,
        add_generation_prompt=True,
    )
    short_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "请用两句话总结什么是向量数据库。"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    events = []
    # step 0: 先来两条长请求，占住 prefill
    events.append((0, long_prompt, "long"))
    events.append((0, long_prompt, "long"))
    # step 10/20/30: 再陆续到达短请求，模拟在线场景
    for s in (10, 20, 30):
        for _ in range(4):
            events.append((s, short_prompt, "short"))
    events.sort(key=lambda x: x[0])
    return events


def run_case(model_path: str, chunk_size: int, batched_tokens: int, enable_continuous_batching: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    events = build_arrival_events(tokenizer)
    sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=192, ignore_eos=True)
    llm = LLM(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_prefill_chunk_size=chunk_size,
        max_num_batched_tokens=batched_tokens,
        enable_continuous_batching=enable_continuous_batching,
    )
    # 预热，避免首次编译/初始化成本污染对比结果。
    llm.generate(["warmup"], SamplingParams(max_tokens=8, ignore_eos=True), use_tqdm=False)

    pending = list(events)
    arrival_time = {}
    seq_group = {}
    first_decode_latency = {}
    finish_latency = {}
    total_output_tokens = 0
    num_steps = 0
    num_mixed_steps = 0

    t0 = perf_counter()
    step_id = 0
    while pending or not llm.is_finished():
        now = perf_counter() - t0
        while pending and pending[0][0] <= step_id:
            _, prompt, group = pending.pop(0)
            seq_id = llm.add_request(prompt, sampling_params)
            arrival_time[seq_id] = now
            seq_group[seq_id] = group

        if llm.is_finished() and not pending:
            break

        outputs, _, _, is_mixed_step, decode_seq_ids = llm.step()
        num_steps += 1
        num_mixed_steps += int(is_mixed_step)
        now = perf_counter() - t0
        for seq_id in decode_seq_ids:
            if seq_id not in first_decode_latency:
                first_decode_latency[seq_id] = now - arrival_time[seq_id]
        for seq_id, token_ids in outputs:
            finish_latency[seq_id] = now - arrival_time[seq_id]
            total_output_tokens += len(token_ids)
        step_id += 1

    short_ttft = [first_decode_latency[s] for s, g in seq_group.items() if g == "short"]
    long_ttft = [first_decode_latency[s] for s, g in seq_group.items() if g == "long"]
    short_finish = [finish_latency[s] for s, g in seq_group.items() if g == "short"]
    long_finish = [finish_latency[s] for s, g in seq_group.items() if g == "long"]
    total_time = perf_counter() - t0

    return {
        "chunk_size": chunk_size,
        "batched_tokens": batched_tokens,
        "total_time": total_time,
        "decode_throughput": total_output_tokens / max(total_time, 1e-6),
        "short_ttft_p95": percentile(short_ttft, 0.95),
        "short_ttft_mean": mean(short_ttft),
        "long_ttft_mean": mean(long_ttft),
        "short_finish_p95": percentile(short_finish, 0.95),
        "short_finish_mean": mean(short_finish),
        "long_finish_mean": mean(long_finish),
        "mixed_ratio": num_mixed_steps / max(1, num_steps),
    }


def aggregate_results(results: list[dict]) -> dict:
    assert results
    keys = results[0].keys()
    agg = {}
    for k in keys:
        values = [r[k] for r in results]
        if isinstance(values[0], (int, float)):
            agg[k] = sum(values) / len(values)
        else:
            agg[k] = values[0]
    return agg


def run_single_case_in_subprocess(model_path: str, chunk_size: int, batched_tokens: int, enable_continuous_batching: bool) -> dict:
    cmd = [
        sys.executable,
        __file__,
        "--mode",
        "single",
        "--model-path",
        model_path,
        "--chunk-size",
        str(chunk_size),
        "--batched-tokens",
        str(batched_tokens),
        "--enable-continuous-batching",
        "1" if enable_continuous_batching else "0",
    ]
    p = subprocess.run(cmd, check=True, capture_output=True, text=True)
    lines = [line.strip() for line in p.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def print_result(name: str, r: dict):
    print(f"\n[{name}]")
    print(f"chunk_size={r['chunk_size']}, max_num_batched_tokens={r['batched_tokens']}")
    print(f"total_time={r['total_time']:.2f}s, decode_throughput={r['decode_throughput']:.2f} tok/s")
    print(f"short_ttft_mean={r['short_ttft_mean']:.2f}s, short_ttft_p95={r['short_ttft_p95']:.2f}s")
    print(f"short_finish_mean={r['short_finish_mean']:.2f}s, short_finish_p95={r['short_finish_p95']:.2f}s")
    print(f"long_ttft_mean={r['long_ttft_mean']:.2f}s, long_finish_mean={r['long_finish_mean']:.2f}s")
    print(f"mixed_ratio={r['mixed_ratio']:.2%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["compare", "single"], default="compare")
    parser.add_argument("--model-path", default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"))
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--batched-tokens", type=int, default=128)
    parser.add_argument("--enable-continuous-batching", type=int, choices=[0, 1], default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--cooldown-seconds", type=float, default=0.0)
    args = parser.parse_args()

    if args.mode == "single":
        result = run_case(
            args.model_path,
            chunk_size=args.chunk_size,
            batched_tokens=args.batched_tokens,
            enable_continuous_batching=bool(args.enable_continuous_batching),
        )
        print(json.dumps(result, ensure_ascii=False))
        return

    baseline_runs = []
    optimized_runs = []
    for i in range(args.repeats):
        if i % 2 == 0:
            baseline_runs.append(
                run_single_case_in_subprocess(
                    args.model_path, chunk_size=4096, batched_tokens=4096, enable_continuous_batching=False
                )
            )
            optimized_runs.append(
                run_single_case_in_subprocess(
                    args.model_path, chunk_size=32, batched_tokens=128, enable_continuous_batching=True
                )
            )
        else:
            optimized_runs.append(
                run_single_case_in_subprocess(
                    args.model_path, chunk_size=32, batched_tokens=128, enable_continuous_batching=True
                )
            )
            baseline_runs.append(
                run_single_case_in_subprocess(
                    args.model_path, chunk_size=4096, batched_tokens=4096, enable_continuous_batching=False
                )
            )
        if args.cooldown_seconds > 0:
            time.sleep(args.cooldown_seconds)
    baseline = aggregate_results(baseline_runs)
    optimized = aggregate_results(optimized_runs)

    print(f"repeats={args.repeats}, cooldown_seconds={args.cooldown_seconds}")
    print_result("baseline_prefill_first_like", baseline)
    print_result("chunked_continuous", optimized)

    print("\n[delta: optimized - baseline]")
    print(f"Δtotal_time={optimized['total_time'] - baseline['total_time']:+.2f}s")
    print(f"Δdecode_throughput={optimized['decode_throughput'] - baseline['decode_throughput']:+.2f} tok/s")
    print(f"Δshort_ttft_p95={optimized['short_ttft_p95'] - baseline['short_ttft_p95']:+.2f}s")
    print(f"Δshort_finish_p95={optimized['short_finish_p95'] - baseline['short_finish_p95']:+.2f}s")
    print(f"Δmixed_ratio={optimized['mixed_ratio'] - baseline['mixed_ratio']:+.2%}")


if __name__ == "__main__":
    main()
