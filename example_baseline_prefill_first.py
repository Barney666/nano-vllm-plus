import os
from time import perf_counter

from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams
from example import build_demo_prompts


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)

    # 近似“未实现 chunked+continuous batching 前”的行为（注意：这是近似，不是完全回退）：
    # - chunk size 足够大，尽量一次性 prefill
    # - batched token budget 放宽，减少 prefill 被切块概率
    llm = LLM(
        path,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_prefill_chunk_size=4096,
        max_num_batched_tokens=4096,
    )

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        max_tokens=384,
        ignore_eos=True,
    )
    prompts = build_demo_prompts(tokenizer)

    t0 = perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = perf_counter() - t0
    stats = llm.last_generate_stats

    print("\n=== Baseline Scheduling Stats (prefill-first like) ===")
    print(f"Total inference time: {elapsed:.2f}s")
    print(f"Total steps: {stats['num_steps']}")
    print(f"Mixed steps (prefill+decode): {stats['num_mixed_steps']}")
    if stats["num_steps"]:
        ratio = stats["num_mixed_steps"] / stats["num_steps"]
        print(f"Mixed ratio: {ratio:.2%}")
    print(f"Prefill tokens scheduled: {stats['prefill_tokens']}")
    print(f"Decode tokens scheduled: {stats['decode_tokens']}")

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
