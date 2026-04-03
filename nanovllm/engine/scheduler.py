from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_prefill_chunk_size = config.max_prefill_chunk_size
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], list[Sequence], list[int]]:
        decode_seqs = []
        prefill_seqs = []
        prefill_chunk_lens = []
        scheduled_ids = set()
        num_batched_tokens = 0

        # decode first
        for seq in list(self.running):
            if len(scheduled_ids) >= self.max_num_seqs or num_batched_tokens >= self.max_num_batched_tokens:
                break
            if not seq.is_prefill_finished:
                continue
            if not self.block_manager.can_append(seq):
                self.preempt(seq)
                continue
            self.block_manager.may_append(seq)
            decode_seqs.append(seq)
            scheduled_ids.add(seq.seq_id)
            num_batched_tokens += 1

        def schedule_prefill(seq: Sequence) -> bool:
            nonlocal num_batched_tokens
            if seq.seq_id in scheduled_ids or seq.is_prefill_finished:
                return False
            if len(scheduled_ids) >= self.max_num_seqs or num_batched_tokens >= self.max_num_batched_tokens:
                return False
            remaining_budget = self.max_num_batched_tokens - num_batched_tokens
            chunk_len = min(seq.remaining_prompt_tokens, self.max_prefill_chunk_size, remaining_budget)
            if chunk_len <= 0:
                return False
            prefill_seqs.append(seq)
            prefill_chunk_lens.append(chunk_len)
            scheduled_ids.add(seq.seq_id)
            num_batched_tokens += chunk_len
            return True

        # continue prefill on running requests first
        for seq in list(self.running):
            if not schedule_prefill(seq):
                continue

        # admit new waiting requests
        while self.waiting and len(scheduled_ids) < self.max_num_seqs and num_batched_tokens < self.max_num_batched_tokens:
            seq = self.waiting[0]
            if not self.block_manager.can_allocate(seq):
                break
            self.block_manager.allocate(seq)
            seq.num_prefilled_tokens = max(seq.num_prefilled_tokens, seq.num_cached_tokens)
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            if schedule_prefill(seq):
                continue
            if not seq.is_prefill_finished or seq.seq_id in scheduled_ids:
                continue
            if len(scheduled_ids) >= self.max_num_seqs or num_batched_tokens >= self.max_num_batched_tokens:
                continue
            if not self.block_manager.can_append(seq):
                self.preempt(seq)
                continue
            self.block_manager.may_append(seq)
            decode_seqs.append(seq)
            scheduled_ids.add(seq.seq_id)
            num_batched_tokens += 1

        if not decode_seqs and not prefill_seqs:
            for seq in list(self.running):
                if seq.seq_id in scheduled_ids or not seq.is_prefill_finished:
                    continue
                if not self.block_manager.can_append(seq):
                    continue
                self.block_manager.may_append(seq)
                decode_seqs.append(seq)
                break
        assert decode_seqs or prefill_seqs
        return decode_seqs, prefill_seqs, prefill_chunk_lens

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        if seq in self.running:
            self.running.remove(seq)
        self.waiting.appendleft(seq)

    def postprocess(self,
                    prefill_seqs: list[Sequence],
                    prefill_chunk_lens: list[int],
                    decode_seqs: list[Sequence],
                    token_ids: list[int]):
        for seq, chunk_len in zip(prefill_seqs, prefill_chunk_lens):
            seq.num_prefilled_tokens = min(seq.num_prompt_tokens, seq.num_prefilled_tokens + chunk_len)
        for seq, token_id in zip(decode_seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
