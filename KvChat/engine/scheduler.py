from collections import deque
import heapq
import logging
import time

from KvChat.config import Config
from KvChat.engine.sequence import Sequence, SequenceStatus
from KvChat.engine.block_manager import BlockManager
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        
        # [OPTIMIZATION] Use heap instead of deque for O(log n) scheduling
        self.waiting_heap = []  # min-heap by sequence length
        self.waiting_dict = {}   # seq_id -> Sequence for fast lookup
        self.running: deque[Sequence] = deque()#!!!
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        
        # Performance monitoring
        self.schedule_stats = {
            'total_calls': 0,
            'total_time_us': 0.0,
            'min_time_us': float('inf'),
            'max_time_us': 0.0,
        }

    def is_finished(self):
        return not self.waiting_dict and not self.running

    def add(self, seq: Sequence):
        # [OPTIMIZATION] Use heap with O(log n) insertion
        heapq.heappush(self.waiting_heap, (len(seq), seq.seq_id, seq))
        self.waiting_dict[seq.seq_id] = seq

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        [OPTIMIZATION] Improved scheduling with O(log n) time complexity
        
        Key improvements:
        1. Use heap to prioritize shorter sequences - O(log n) instead of O(n)
        2. Better GPU resource utilization
        3. Significantly reduced scheduling overhead
        """
        start_time = time.perf_counter()
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        
        # Phase 1: Prefill phase using heap (O(log n) per sequence)
        # Prioritize shorter sequences to reduce latency
        while self.waiting_heap and num_seqs < self.max_num_seqs:
            # Pop shortest sequence from heap - O(log n)
            length, seq_id, seq = heapq.heappop(self.waiting_heap)
            
            # Remove from dict
            if seq_id in self.waiting_dict:
                del self.waiting_dict[seq_id]
            
            if num_seqs >= self.max_num_seqs:
                break
            if num_batched_tokens + length > self.max_num_batched_tokens:
                # Put back to heap if exceeds token limit
                heapq.heappush(self.waiting_heap, (length, seq_id, seq))
                self.waiting_dict[seq_id] = seq
                break
            if not self.block_manager.can_allocate(seq):
                # Put back to heap if can't allocate blocks
                heapq.heappush(self.waiting_heap, (length, seq_id, seq))
                self.waiting_dict[seq_id] = seq
                continue
            
            # Allocate and schedule
            self.block_manager.allocate(seq)
            num_seqs += 1
            num_batched_tokens += length - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            scheduled_seqs.append(seq)
        
        if scheduled_seqs:
            elapsed = (time.perf_counter() - start_time) * 1e6
            self._record_schedule_time(elapsed)
            return scheduled_seqs, True  # Prefill mode

        # Phase 2: Decode phase
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            
            can_allocate = True
            while not self.block_manager.can_append(seq):
                can_allocate = False
                
                if self.running:
                    preempt_seq = self.running.pop()
                    self.preempt(preempt_seq)
                else:
                    self.preempt(seq)
                    break
            
            if can_allocate or self.block_manager.can_append(seq):
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
            else:
                self.running.appendleft(seq)
                break
        
        if not scheduled_seqs and self.running:
            seq = self.running.popleft()
            self.block_manager.may_append(seq)
            scheduled_seqs.append(seq)
            elapsed = (time.perf_counter() - start_time) * 1e6
            self._record_schedule_time(elapsed)
            return scheduled_seqs, False
        elif scheduled_seqs:
            self.running.extendleft(reversed(scheduled_seqs))
            elapsed = (time.perf_counter() - start_time) * 1e6
            self._record_schedule_time(elapsed)
            return scheduled_seqs, False
        else:
            elapsed = (time.perf_counter() - start_time) * 1e6
            self._record_schedule_time(elapsed)
            return [], False
    
    def _record_schedule_time(self, elapsed_us):
        """Record scheduling performance metrics"""
        stats = self.schedule_stats
        stats['total_calls'] += 1
        stats['total_time_us'] += elapsed_us
        stats['min_time_us'] = min(stats['min_time_us'], elapsed_us)
        stats['max_time_us'] = max(stats['max_time_us'], elapsed_us)

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING

        # restore logical sequence length before recompute
        seq.num_tokens = len(seq.token_ids)
        if len(seq.token_ids) > 0:
            seq.last_token = seq.token_ids[-1]
        self.block_manager.deallocate(seq)
        
        # [OPTIMIZATION] Add back to waiting heap for O(log n) reinsertion
        heapq.heappush(self.waiting_heap, (len(seq), seq.seq_id, seq))
        self.waiting_dict[seq.seq_id] = seq

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], compression_events: list | None = None) -> list[bool]:

        if compression_events:
            #if compressed, we need to free blocks beyond R (truncate_blocks())
            #and set seq.num_tokens as R
            #Note: we use "rope_pos" in model_runner.py variable to denote positions variable RoPE, which is the actual seq len. 
            for ev in compression_events:
                bidx = ev["batch_index"]
                if bidx < 0 or bidx >= len(seqs):
                    continue
                seq = seqs[bidx]
                R = ev["R"]
                keep_blocks = ev["keep_blocks"]
                self.block_manager.truncate_blocks(seq, keep_blocks)
                seq.num_tokens = R

        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            # print(self.tokenizer.decode(seq.token_ids))
            if (not seq.ignore_eos and token_id == self.eos) or seq.generated_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
