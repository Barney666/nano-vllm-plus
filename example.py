import os
from time import perf_counter
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def build_long_context_prompt() -> str:
    # 构造一个更贴近真实业务的长输入：企业知识库检索 + 多份项目周报摘要。
    header = (
        "你是企业内部项目助理。下面给你一批来自不同团队的周报摘要、风险记录和行动项，"
        "请在阅读全部材料后输出：1) 关键风险Top5；2) 下周优先级最高的行动计划；3) 给管理层的简短结论。"
    )
    sections = []
    for i in range(1, 41):
        sections.append(
            f"[周报#{i}] 团队=数据平台, 主题=日志治理与成本优化, "
            f"本周进展=完成S{i:02d}批次指标口径对齐并上线告警, "
            f"阻塞问题=上游埋点遗漏字段导致回填延迟{i%7+1}小时, "
            f"风险等级={'高' if i % 9 == 0 else '中' if i % 4 == 0 else '低'}, "
            f"下周计划=推进跨部门Schema评审与历史分区重算。"
        )
    return header + "\n\n" + "\n".join(sections)


def build_demo_prompts(tokenizer: AutoTokenizer) -> list[str]:
    prompts = [
        build_long_context_prompt(),  # 用一个超长 prompt 制造持续 prefill 压力
        "请用三句话介绍你自己。",
        "列出100以内所有质数，并按逗号分隔。",
        "写一首关于并行计算和缓存复用的短诗。",
        "给我一个Python函数，输入n返回斐波那契数列前n项。",
        "请用通俗语言解释连续批处理（continuous batching）。",
    ]
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]


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

    print("\n=== Scheduling Stats ===")
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
