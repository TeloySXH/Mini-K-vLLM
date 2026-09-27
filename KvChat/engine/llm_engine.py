import atexit
from dataclasses import fields
from time import perf_counter
from typing import Dict
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp
import torch
import logging
from KvChat.config import Config
from KvChat.sampling_params import SamplingParams
from KvChat.engine.sequence import Sequence
from KvChat.engine.scheduler import Scheduler
from KvChat.engine.model_runner import ModelRunner
from KvChat.utils.perf_monitor import get_perf_monitor

logger = logging.getLogger(__name__)


def worker_process(config: Config, rank: int, event):
    """
    Worker process function for distributed inference.
    Initializes ModelRunner and enters event loop.
    """
    import sys
    import os
    
    # Configure output buffering for proper logging
    sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)  # Line buffering
    sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)
    
    model_runner = ModelRunner(config, rank, event)
    # Enter the event loop for this worker
    if model_runner.world_size > 1 and model_runner.rank > 0:
        model_runner.loop()


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        # Create worker processes for ranks 1 to world_size-1
        for i in range(1, config.world_size):
            event = ctx.Event()
            process = ctx.Process(target=worker_process, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        # Initialize master rank (rank 0) ModelRunner
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        
        # [OPTIMIZATION] Enhanced performance monitoring (P2.4)
        self.perf_monitor = get_perf_monitor()
        self.session_stats = {
            'total_prefill_tokens': 0,
            'total_decode_tokens': 0,
            'total_prefill_time_ms': 0.0,
            'total_decode_time_ms': 0.0,
            'prefill_count': 0,
            'decode_count': 0,
            'max_batch_size': 0,
            'num_sequences': 0,
            'peak_memory_mb': 0.0,
        }
        
        atexit.register(self.exit)

    def exit(self):
        if hasattr(self, 'model_runner'):
            self.model_runner.call("exit")
            del self.model_runner
        if hasattr(self, 'ps'):
            for p in self.ps:
                p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        
        # [OPTIMIZATION] Track performance metrics (P2.4)
        perf_monitor = get_perf_monitor()
        timer_name = f"{'prefill' if is_prefill else 'decode'}_kernel"
        perf_monitor.start_timer(timer_name)
        
        self.model_runner.call("set_compress_enabled", self.model_runner.config.kv_compress_enabled)
        
        ret = self.model_runner.call("run", seqs, is_prefill)
        if isinstance(ret, tuple):
            token_ids, compression_events = ret
        else:
            token_ids, compression_events = ret, None
        
        # [OPTIMIZATION] Record timing
        elapsed_ms = perf_monitor.end_timer(timer_name)

        self.scheduler.postprocess(seqs, token_ids, compression_events)
        
        # [OPTIMIZATION] Update session statistics
        if is_prefill:
            self.session_stats['total_prefill_tokens'] += sum(len(seq) for seq in seqs)
            self.session_stats['total_prefill_time_ms'] += elapsed_ms
            self.session_stats['prefill_count'] += 1
        else:
            self.session_stats['total_decode_tokens'] += len(seqs)
            self.session_stats['total_decode_time_ms'] += elapsed_ms
            self.session_stats['decode_count'] += 1
        
        self.session_stats['max_batch_size'] = max(self.session_stats['max_batch_size'], len(seqs))
        self.session_stats['num_sequences'] = len(seqs)

        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)

        # 新增：把本轮 decode 新生成的 token_ids 一并返回
        step_token_ids = token_ids if not is_prefill else None

        return outputs, num_tokens, step_token_ids, None, None


    def is_finished(self):
        return self.scheduler.is_finished()
    
    def get_session_stats(self) -> Dict:
        """
        [OPTIMIZATION] Get comprehensive session performance statistics (P2.4)
        
        Returns:
            Dict with:
            - throughput metrics (tokens/sec)
            - latency breakdown (prefill vs decode)
            - memory metrics
            - batch efficiency
        """
        try:
            # Get memory stats
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                memory_reserved = torch.cuda.memory_reserved() / 1e9
                memory_allocated = torch.cuda.memory_allocated() / 1e9
            else:
                memory_reserved = 0.0
                memory_allocated = 0.0
            
            # Calculate throughput
            total_time_ms = self.session_stats['total_prefill_time_ms'] + self.session_stats['total_decode_time_ms']
            total_tokens = self.session_stats['total_prefill_tokens'] + self.session_stats['total_decode_tokens']
            overall_throughput = (total_tokens / total_time_ms * 1000) if total_time_ms > 0 else 0.0
            
            prefill_throughput = 0.0
            if self.session_stats['total_prefill_time_ms'] > 0:
                prefill_throughput = self.session_stats['total_prefill_tokens'] / (self.session_stats['total_prefill_time_ms'] / 1000)
            
            decode_throughput = 0.0
            if self.session_stats['total_decode_time_ms'] > 0:
                decode_throughput = self.session_stats['total_decode_tokens'] / (self.session_stats['total_decode_time_ms'] / 1000)
            
            return {
                'overall_throughput_tok_per_sec': overall_throughput,
                'prefill_throughput_tok_per_sec': prefill_throughput,
                'decode_throughput_tok_per_sec': decode_throughput,
                'total_prefill_tokens': self.session_stats['total_prefill_tokens'],
                'total_decode_tokens': self.session_stats['total_decode_tokens'],
                'total_prefill_time_ms': self.session_stats['total_prefill_time_ms'],
                'total_decode_time_ms': self.session_stats['total_decode_time_ms'],
                'prefill_count': self.session_stats['prefill_count'],
                'decode_count': self.session_stats['decode_count'],
                'max_batch_size': self.session_stats['max_batch_size'],
                'peak_memory_reserved_gb': memory_reserved,
                'current_memory_allocated_gb': memory_allocated,
                'num_sequences_processed': self.session_stats['num_sequences'],
            }
        except Exception as e:
            logger.warning(f"[LLMEngine] Error collecting session stats: {e}")
            return {}
    
    def print_session_stats(self):
        """Print formatted session statistics."""
        stats = self.get_session_stats()
        
        print("\n" + "="*70)
        print("Session Statistics")
        print("="*70)
        print(f"Overall Throughput:        {stats.get('overall_throughput_tok_per_sec', 0):.2f} tok/s")
        print(f"  Prefill Phase:           {stats.get('prefill_throughput_tok_per_sec', 0):.2f} tok/s")
        print(f"  Decode Phase:            {stats.get('decode_throughput_tok_per_sec', 0):.2f} tok/s")
        print(f"\nToken Statistics:")
        print(f"  Total Prefill Tokens:    {stats.get('total_prefill_tokens', 0)}")
        print(f"  Total Decode Tokens:     {stats.get('total_decode_tokens', 0)}")
        print(f"  Total Time (ms):         {stats.get('total_prefill_time_ms', 0) + stats.get('total_decode_time_ms', 0):.2f}")
        print(f"\nBatch Efficiency:")
        print(f"  Max Batch Size:          {stats.get('max_batch_size', 0)}")
        print(f"  Sequences Processed:     {stats.get('num_sequences_processed', 0)}")
        print(f"\nMemory Usage:")
        print(f"  Peak GPU Memory:         {stats.get('peak_memory_reserved_gb', 0):.2f} GB")
        print(f"  Current Allocation:      {stats.get('current_memory_allocated_gb', 0):.2f} GB")
        print("="*70 + "\n")

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.

        decode_tp_sum = 0.0
        decode_tp_steps = 0

        while not self.is_finished():
            torch.cuda.synchronize()
            t = perf_counter()
            output, num_tokens, _ = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    torch.cuda.synchronize()
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    torch.cuda.synchronize()
                    decode_throughput = -num_tokens / (perf_counter() - t)


                # 新增：只在 decode step 累加
                decode_tp_sum += decode_throughput
                decode_tp_steps += 1

                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        avg_decode_tp = decode_tp_sum / decode_tp_steps if decode_tp_steps > 0 else 0.0
        # print(f"[Metrics] Avg Decode Throughput (step mean): {avg_decode_tp:.2f} tok/s "
        #     f"over {decode_tp_steps} decode steps")
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
