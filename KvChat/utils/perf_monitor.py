"""
Performance monitoring utilities for LLM inference.
Tracks token throughput, latency, and kernel timing.
"""

import time
from collections import defaultdict
from typing import Dict, List
import torch


class PerformanceMonitor:
    """Monitor and track inference performance metrics."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.timers = {}
        self.kernel_times = defaultdict(float)
        
    def start_timer(self, name: str):
        """Start a named timer."""
        torch.cuda.synchronize()
        self.timers[name] = time.perf_counter()
    
    def end_timer(self, name: str) -> float:
        """End a named timer and return elapsed time in milliseconds."""
        torch.cuda.synchronize()
        if name not in self.timers:
            return 0
        elapsed = (time.perf_counter() - self.timers[name]) * 1000
        self.kernel_times[name] += elapsed
        return elapsed
    
    def record_prefill(self, num_tokens: int, latency_ms: float):
        """Record prefill phase metrics."""
        throughput = num_tokens / (latency_ms / 1000) if latency_ms > 0 else 0
        self.metrics['prefill_throughput'].append(throughput)
        self.metrics['prefill_latency'].append(latency_ms)
    
    def record_decode(self, num_tokens: int, latency_ms: float):
        """Record decode phase metrics."""
        throughput = num_tokens / (latency_ms / 1000) if latency_ms > 0 else 0
        self.metrics['decode_throughput'].append(throughput)
        self.metrics['decode_latency'].append(latency_ms)
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        summary = {}
        
        for metric_name, values in self.metrics.items():
            if values:
                summary[f"{metric_name}_avg"] = sum(values) / len(values)
                summary[f"{metric_name}_max"] = max(values)
                summary[f"{metric_name}_min"] = min(values)
        
        for kernel_name, total_time in self.kernel_times.items():
            summary[f"kernel_{kernel_name}_total_ms"] = total_time
        
        return summary
    
    def print_summary(self):
        """Print performance summary."""
        summary = self.get_summary()
        print("\n" + "="*60)
        print("Performance Summary")
        print("="*60)
        
        if 'prefill_throughput_avg' in summary:
            print(f"Prefill Throughput: {summary['prefill_throughput_avg']:.1f} tok/s")
        
        if 'decode_throughput_avg' in summary:
            print(f"Decode Throughput:  {summary['decode_throughput_avg']:.1f} tok/s")
        
        print("\nKernel Timing Breakdown:")
        for kernel_name, total_time in self.kernel_times.items():
            print(f"  {kernel_name}: {total_time:.2f}ms")
        
        print("="*60 + "\n")
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        self.kernel_times.clear()
        self.timers.clear()


# Global performance monitor instance
_perf_monitor = None


def get_perf_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _perf_monitor
    if _perf_monitor is None:
        _perf_monitor = PerformanceMonitor()
    return _perf_monitor
