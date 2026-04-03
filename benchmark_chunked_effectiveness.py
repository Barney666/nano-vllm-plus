import argparse
import json
import os
import subprocess
import sys
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


def build_eval_prompts(tokenizer) -> tuple[list[str], list[str]]:
    long_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": build_long_context_prompt()}],
        tokenize=False,
        add_generation_prompt=True,
    )
    short_raw = [
        "请用三句话解释什么是RAG。",
        "给一个Python列表去重的示例。",
        "帮我写一段会议纪要开头。",
        "解释一下Top-p采样。",
        "什么是KV cache？",
    ]
    short_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in short_raw
    ]
    prompts = [long_prompt] * 4 + short_prompts * 4
    groups = ["long"] * 4 + ["short"] * (len(short_prompts) * 4)
    return prompts, groups


def run_case(model_path: str, chunk_size: int, batched_tokens: int):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompts, groups = build_eval_prompts(tokenizer)
    sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=192, ignore_eos=True)
    llm = LLM(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_prefill_chunk_size=chunk_size,
        max_num_batched_tokens=batched_tokens,
    )

    seq_group = {}
    first_decode_time = {}
    finish_time = {}
    total_output_tokens = 0
    num_steps = 0
    num_mixed_steps = 0

    t0 = perf_counter()
    for prompt, group in zip(prompts, groups):
        seq_id = llm.add_request(prompt, sampling_params)
        seq_group[seq_id] = group

    while not llm.is_finished():
        outputs, _, _, is_mixed_step, decode_seq_ids = llm.step()
        num_steps += 1
        num_mixed_steps += int(is_mixed_step)
        now = perf_counter() - t0
        for seq_id in decode_seq_ids:
            first_decode_time.setdefault(seq_id, now)
        for seq_id, token_ids in outputs:
            finish_time[seq_id] = now
            total_output_tokens += len(token_ids)

    total_time = perf_counter() - t0

    ttft_all = [first_decode_time[s] for s in seq_group]
    ttft_short = [first_decode_time[s] for s, g in seq_group.items() if g == "short"]
    ttft_long = [first_decode_time[s] for s, g in seq_group.items() if g == "long"]
    t_finish_all = [finish_time[s] for s in seq_group]
    t_finish_short = [finish_time[s] for s, g in seq_group.items() if g == "short"]
    t_finish_long = [finish_time[s] for s, g in seq_group.items() if g == "long"]

    return {
        "chunk_size": chunk_size,
        "batched_tokens": batched_tokens,
        "total_time": total_time,
        "decode_throughput": total_output_tokens / max(total_time, 1e-6),
        "ttft_p50": percentile(ttft_all, 0.5),
        "ttft_p95": percentile(ttft_all, 0.95),
        "ttft_short_mean": mean(ttft_short),
        "ttft_long_mean": mean(ttft_long),
        "finish_p95": percentile(t_finish_all, 0.95),
        "finish_short_mean": mean(t_finish_short),
        "finish_long_mean": mean(t_finish_long),
        "mixed_ratio": num_mixed_steps / max(1, num_steps),
    }


def print_result(name: str, r: dict):
    print(f"\n[{name}]")
    print(f"chunk_size={r['chunk_size']}, max_num_batched_tokens={r['batched_tokens']}")
    print(f"total_time={r['total_time']:.2f}s")
    print(f"decode_throughput={r['decode_throughput']:.2f} tok/s")
    print(f"ttft_p50={r['ttft_p50']:.2f}s, ttft_p95={r['ttft_p95']:.2f}s")
    print(f"ttft_short_mean={r['ttft_short_mean']:.2f}s, ttft_long_mean={r['ttft_long_mean']:.2f}s")
    print(f"finish_p95={r['finish_p95']:.2f}s")
    print(f"finish_short_mean={r['finish_short_mean']:.2f}s, finish_long_mean={r['finish_long_mean']:.2f}s")
    print(f"mixed_ratio={r['mixed_ratio']:.2%}")


def run_single_case_in_subprocess(model_path: str, chunk_size: int, batched_tokens: int) -> dict:
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
    ]
    p = subprocess.run(cmd, check=True, capture_output=True, text=True)
    lines = [line.strip() for line in p.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["compare", "single"], default="compare")
    parser.add_argument("--model-path", default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"))
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--batched-tokens", type=int, default=128)
    args = parser.parse_args()

    if args.mode == "single":
        result = run_case(args.model_path, chunk_size=args.chunk_size, batched_tokens=args.batched_tokens)
        print(json.dumps(result, ensure_ascii=False))
        return

    baseline = run_single_case_in_subprocess(args.model_path, chunk_size=4096, batched_tokens=4096)
    optimized = run_single_case_in_subprocess(args.model_path, chunk_size=32, batched_tokens=128)

    print_result("baseline_prefill_first_like", baseline)
    print_result("chunked_continuous", optimized)

    print("\n[delta: optimized - baseline]")
    print(f"Δtotal_time={optimized['total_time'] - baseline['total_time']:+.2f}s")
    print(f"Δdecode_throughput={optimized['decode_throughput'] - baseline['decode_throughput']:+.2f} tok/s")
    print(f"Δttft_p95={optimized['ttft_p95'] - baseline['ttft_p95']:+.2f}s")
    print(f"Δttft_short_mean={optimized['ttft_short_mean'] - baseline['ttft_short_mean']:+.2f}s")
    print(f"Δfinish_p95={optimized['finish_p95'] - baseline['finish_p95']:+.2f}s")
    print(f"Δfinish_short_mean={optimized['finish_short_mean'] - baseline['finish_short_mean']:+.2f}s")
    print(f"Δmixed_ratio={optimized['mixed_ratio'] - baseline['mixed_ratio']:+.2%}")


if __name__ == "__main__":
    main()
