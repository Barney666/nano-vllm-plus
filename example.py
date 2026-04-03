import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(
        path,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_prefill_chunk_size=32,
        max_num_batched_tokens=128,
    )

    sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=128)
    long_prompt = "请详细解释Transformer中的注意力机制。"
    long_prompt += "请分点说明并给出一个简单例子。" * 256
    prompts = [
        long_prompt,  # 用一个超长 prompt 制造持续 prefill 压力
        "请用三句话介绍你自己。",
        "列出100以内所有质数，并按逗号分隔。",
        "写一首关于并行计算和缓存复用的短诗。",
        "给我一个Python函数，输入n返回斐波那契数列前n项。",
        "请用通俗语言解释连续批处理（continuous batching）。",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)
    stats = llm.last_generate_stats

    print("\n=== Scheduling Stats ===")
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
