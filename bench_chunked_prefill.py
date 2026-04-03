import argparse
import os
import random
from time import perf_counter

import torch

from nanovllm import LLM, SamplingParams


def _find_seq(engine, seq_id):
    for seq in engine.scheduler.running:
        if seq.seq_id == seq_id:
            return seq
    for seq in engine.scheduler.waiting:
        if seq.seq_id == seq_id:
            return seq
    return None


def run_case(model_path: str, enable_chunked_prefill: bool, chunk_size: int):
    llm = LLM(
        model_path,
        enforce_eager=True,
        max_model_len=32768,
        max_num_batched_tokens=32768,
        max_num_seqs=64,
        enable_chunked_prefill=enable_chunked_prefill,
        chunked_prefill_size=chunk_size,
    )

    # warmup
    llm.generate([[1, 2, 3, 4]], SamplingParams(max_tokens=1), use_tqdm=False)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    random.seed(0)
    long_prompt = [random.randint(0, 10000) for _ in range(32768)]
    short_prompts = [[random.randint(0, 10000) for _ in range(128)] for _ in range(32)]

    sampling = SamplingParams(temperature=0.0, ignore_eos=True, max_tokens=32)
    long_seq_id = llm.add_request(long_prompt, sampling)
    short_seq_ids = [llm.add_request(prompt, sampling) for prompt in short_prompts]

    start = perf_counter()
    ttft = {}
    done = set()
    long_done_time = None

    while not llm.is_finished():
        llm.step()
        elapsed_ms = (perf_counter() - start) * 1000

        for seq_id in [long_seq_id, *short_seq_ids]:
            if seq_id in ttft:
                continue
            seq = _find_seq(llm, seq_id)
            if seq is not None and seq.num_completion_tokens > 0:
                ttft[seq_id] = elapsed_ms

        for seq_id in [long_seq_id, *short_seq_ids]:
            if seq_id in done:
                continue
            seq = _find_seq(llm, seq_id)
            if seq is None:
                done.add(seq_id)
                if seq_id == long_seq_id and long_done_time is None:
                    long_done_time = elapsed_ms

    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    llm.exit()

    long_ttft = ttft[long_seq_id]
    short_ttfts = [ttft[seq_id] for seq_id in short_seq_ids]
    blocking_rate = sum(value > long_ttft for value in short_ttfts) / len(short_ttfts)

    return {
        "mode": "Chunked Prefill" if enable_chunked_prefill else "Full Prefill",
        "chunk_size": chunk_size if enable_chunked_prefill else "-",
        "long_ttft_ms": long_ttft,
        "short_ttft_avg_ms": sum(short_ttfts) / len(short_ttfts),
        "peak_memory_gb": peak_gb,
        "head_blocking_rate": blocking_rate,
        "long_done_ms": long_done_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare Full Prefill vs Chunked Prefill")
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"))
    parser.add_argument("--chunk-size", type=int, default=2048)
    args = parser.parse_args()

    baseline = run_case(args.model, enable_chunked_prefill=False, chunk_size=args.chunk_size)
    chunked = run_case(args.model, enable_chunked_prefill=True, chunk_size=args.chunk_size)

    print("\n=== Chunked Prefill Benchmark (1 x 32K + 32 x short) ===")
    print("| Mode | Chunk Size | Long TTFT (ms) | Short TTFT Avg (ms) | Peak Memory (GB) | Head Blocking Rate | Long Done (ms) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in [baseline, chunked]:
        print(
            f"| {row['mode']} | {row['chunk_size']} | {row['long_ttft_ms']:.2f} | "
            f"{row['short_ttft_avg_ms']:.2f} | {row['peak_memory_gb']:.2f} | "
            f"{row['head_blocking_rate']:.2%} | {row['long_done_ms']:.2f} |"
        )


if __name__ == "__main__":
    main()
