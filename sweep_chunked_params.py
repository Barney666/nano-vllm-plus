import os
from time import perf_counter

from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams
from example import build_demo_prompts


def run_once(model_path: str, chunk_size: int, batched_tokens: int):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompts = build_demo_prompts(tokenizer)
    sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=384, ignore_eos=True)
    llm = LLM(
        model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_prefill_chunk_size=chunk_size,
        max_num_batched_tokens=batched_tokens,
    )
    t0 = perf_counter()
    llm.generate(prompts, sampling_params, use_tqdm=False)
    elapsed = perf_counter() - t0
    stats = llm.last_generate_stats
    return {
        "chunk": chunk_size,
        "batched": batched_tokens,
        "time": elapsed,
        "mixed_ratio": stats["num_mixed_steps"] / max(1, stats["num_steps"]),
        "steps": stats["num_steps"],
    }


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    candidates = [
        (32, 128),
        (64, 256),
        (128, 512),
        (256, 1024),
        (4096, 4096),  # baseline-like
    ]
    print("chunk_size | max_num_batched_tokens | time(s) | mixed_ratio | steps")
    print("-" * 72)
    for chunk_size, batched_tokens in candidates:
        r = run_once(path, chunk_size, batched_tokens)
        print(f"{r['chunk']:>10} | {r['batched']:>22} | {r['time']:>7.2f} | {r['mixed_ratio']:.2%:>10} | {r['steps']:>5}")


if __name__ == "__main__":
    main()
