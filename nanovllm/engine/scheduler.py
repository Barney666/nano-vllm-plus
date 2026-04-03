from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.enable_chunked_prefill = config.enable_chunked_prefill
        self.chunked_prefill_size = config.chunked_prefill_size
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool, list[bool]]:
        if self.enable_chunked_prefill:
            return self._schedule_chunked_prefill()

        return self._schedule_full_prefill()

    def _schedule_full_prefill(self) -> tuple[list[Sequence], bool, list[bool]]:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True, [True] * len(scheduled_seqs)

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False, [True] * len(scheduled_seqs)

    def _schedule_chunked_prefill(self) -> tuple[list[Sequence], bool, list[bool]]:
        while self.waiting and len(self.running) < self.max_num_seqs:
            seq = self.waiting[0]
            if not self.block_manager.can_allocate(seq):
                break
            self.block_manager.allocate(seq)
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)

        # prioritize decode to protect short request latency.
        decode_seqs = []
        for _ in range(len(self.running)):
            if len(decode_seqs) >= self.max_num_seqs:
                break
            seq = self.running.popleft()
            if seq.num_cached_tokens < seq.num_prompt_tokens:
                self.running.append(seq)
                continue
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                self.block_manager.may_append(seq)
                decode_seqs.append(seq)
        if decode_seqs:
            self.running.extendleft(reversed(decode_seqs))
            return decode_seqs, False, [True] * len(decode_seqs)

        # schedule prefill chunks.
        prefill_seqs = []
        sample_flags = []
        num_batched_tokens = 0
        for seq in self.running:
            remaining = seq.num_prompt_tokens - seq.num_cached_tokens
            if remaining <= 0:
                continue
            if num_batched_tokens >= self.max_num_batched_tokens:
                break
            chunk_tokens = min(remaining, self.chunked_prefill_size, self.max_num_batched_tokens - num_batched_tokens)
            if chunk_tokens <= 0:
                break
            seq.scheduled_prefill_tokens = chunk_tokens
            num_batched_tokens += chunk_tokens
            prefill_seqs.append(seq)
            sample_flags.append(seq.num_cached_tokens + chunk_tokens == seq.num_prompt_tokens)
            if len(prefill_seqs) >= self.max_num_seqs:
                break
        assert prefill_seqs
        return prefill_seqs, True, sample_flags

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool, sample_flags: list[bool]):
        token_ids = iter(token_ids)
        for seq, sample_token in zip(seqs, sample_flags):
            if is_prefill:
                chunk_tokens = getattr(seq, "scheduled_prefill_tokens", seq.num_prompt_tokens - seq.num_cached_tokens)
                seq.num_cached_tokens += chunk_tokens
                if hasattr(seq, "scheduled_prefill_tokens"):
                    delattr(seq, "scheduled_prefill_tokens")
            if not sample_token:
                continue
            token_id = next(token_ids)
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
