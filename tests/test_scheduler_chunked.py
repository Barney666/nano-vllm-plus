import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[1]
NANOVLLM_ROOT = ROOT / "nanovllm"

if "nanovllm" not in sys.modules:
    pkg = types.ModuleType("nanovllm")
    pkg.__path__ = [str(NANOVLLM_ROOT)]
    sys.modules["nanovllm"] = pkg
if "nanovllm.engine" not in sys.modules:
    pkg = types.ModuleType("nanovllm.engine")
    pkg.__path__ = [str(NANOVLLM_ROOT / "engine")]
    sys.modules["nanovllm.engine"] = pkg
if "nanovllm.config" not in sys.modules:
    mod = types.ModuleType("nanovllm.config")
    mod.Config = object
    sys.modules["nanovllm.config"] = mod
if "xxhash" not in sys.modules:
    class _FakeXXH64:
        def __init__(self):
            self._data = bytearray()

        def update(self, data: bytes):
            self._data.extend(data)

        def intdigest(self):
            return hash(bytes(self._data)) & ((1 << 64) - 1)

    mod = types.ModuleType("xxhash")
    mod.xxh64 = _FakeXXH64
    sys.modules["xxhash"] = mod
if "numpy" not in sys.modules:
    class _FakeArray:
        def __init__(self, values):
            self._values = values

        def tobytes(self):
            return b"".join(int(v).to_bytes(8, "little", signed=True) for v in self._values)

    mod = types.ModuleType("numpy")
    mod.array = _FakeArray
    sys.modules["numpy"] = mod

load_module("nanovllm.sampling_params", NANOVLLM_ROOT / "sampling_params.py")
load_module("nanovllm.engine.sequence", NANOVLLM_ROOT / "engine" / "sequence.py")
block_manager_mod = load_module("nanovllm.engine.block_manager", NANOVLLM_ROOT / "engine" / "block_manager.py")
scheduler_mod = load_module("nanovllm.engine.scheduler", NANOVLLM_ROOT / "engine" / "scheduler.py")
sequence_mod = sys.modules["nanovllm.engine.sequence"]
Scheduler = scheduler_mod.Scheduler
Sequence = sequence_mod.Sequence
BlockManager = block_manager_mod.BlockManager


def make_scheduler(max_num_batched_tokens=8, max_num_seqs=8, max_prefill_chunk_size=3):
    cfg = SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_prefill_chunk_size=max_prefill_chunk_size,
        eos=-1,
        num_kvcache_blocks=128,
        kvcache_block_size=256,
    )
    return Scheduler(cfg)


def test_mixed_decode_and_chunked_prefill_in_same_step():
    scheduler = make_scheduler(max_num_batched_tokens=8, max_prefill_chunk_size=3)

    seq1 = Sequence([1, 2, 3, 4, 5])
    scheduler.add(seq1)

    decode, prefill, chunks = scheduler.schedule()
    assert decode == []
    assert prefill == [seq1]
    assert chunks == [3]
    scheduler.postprocess(prefill, chunks, decode, [])
    assert seq1.num_prefilled_tokens == 3

    seq2 = Sequence([9, 8, 7, 6])
    scheduler.add(seq2)

    decode, prefill, chunks = scheduler.schedule()
    assert decode == []
    assert prefill == [seq1, seq2]
    assert chunks == [2, 3]
    scheduler.postprocess(prefill, chunks, decode, [])
    assert seq1.is_prefill_finished
    assert seq2.num_prefilled_tokens == 3

    decode, prefill, chunks = scheduler.schedule()
    assert decode == [seq1]
    assert prefill == [seq2]
    assert chunks == [1]


def test_prefill_finished_waiting_sequence_can_decode_same_step():
    scheduler = make_scheduler(max_num_batched_tokens=8, max_prefill_chunk_size=3)

    seq = Sequence([1, 2, 3, 4, 5])
    seq.num_prefilled_tokens = seq.num_prompt_tokens
    scheduler.add(seq)

    decode, prefill, chunks = scheduler.schedule()
    assert prefill == []
    assert chunks == []
    assert decode == [seq]


def test_block_manager_appends_new_block_at_block_boundary():
    old_block_size = Sequence.block_size
    Sequence.block_size = 4
    try:
        seq = Sequence([1, 2, 3, 4])
        bm = BlockManager(num_blocks=16, block_size=4)
        bm.allocate(seq)
        assert bm.can_append(seq)
        bm.may_append(seq)
        assert len(seq.block_table) == 2
    finally:
        Sequence.block_size = old_block_size


def test_schedule_respects_token_budget_and_no_duplicate_seq_in_step():
    scheduler = make_scheduler(max_num_batched_tokens=4, max_prefill_chunk_size=3)

    seq1 = Sequence([1, 2, 3, 4, 5])  # first step chunk 3, remaining 2
    seq2 = Sequence([6, 7, 8, 9, 10])  # should be capped by remaining budget
    scheduler.add(seq1)
    scheduler.add(seq2)

    decode, prefill, chunks = scheduler.schedule()
    assert decode == []
    assert prefill == [seq1, seq2]
    assert chunks == [3, 1]
    assert sum(chunks) <= 4
    assert len({seq.seq_id for seq in prefill + decode}) == len(prefill) + len(decode)
