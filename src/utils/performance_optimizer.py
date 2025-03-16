#!/usr/bin/env python3
"""
Performance optimization utilities for VSAT.

This module provides tools and utilities for optimizing the performance
of VSAT, particularly when processing large audio files.
"""

import os
import logging
import time
import psutil
import threading
import numpy as np
from typing import Dict, Any, Optional, Callable, List, Tuple
from functools import wraps

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor and track performance metrics during processing."""
    
    def __init__(self):
        """Initialize the performance monitor."""
        self.metrics = {}
        self.start_times = {}
        self.process = psutil.Process(os.getpid())
        self.baseline_memory = self.get_memory_usage()
        self.max_memory = self.baseline_memory
        self.monitoring_active = False
        self.monitor_thread = None
        self.lock = threading.Lock()
    
    def start_monitoring(self, interval: float = 1.0):
        """Start continuous monitoring in a background thread.
        
        Args:
            interval: Time between measurements in seconds
        """
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.debug("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop the continuous monitoring thread."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
            self.monitor_thread = None
        logger.debug("Performance monitoring stopped")
    
    def _monitor_loop(self, interval: float):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                memory = self.get_memory_usage()
                with self.lock:
                    if memory > self.max_memory:
                        self.max_memory = memory
                
                # Sleep for the specified interval
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error in performance monitoring: {str(e)}")
                time.sleep(interval)  # Continue monitoring despite errors
    
    def start_timer(self, name: str):
        """Start a timer for a specific operation.
        
        Args:
            name: Name of the operation to time
        """
        self.start_times[name] = time.time()
    
    def stop_timer(self, name: str) -> float:
        """Stop a timer and record the elapsed time.
        
        Args:
            name: Name of the operation to stop timing
            
        Returns:
            Elapsed time in seconds
        """
        if name not in self.start_times:
            logger.warning(f"Timer '{name}' was never started")
            return 0.0
        
        elapsed = time.time() - self.start_times[name]
        
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = {"count": 0, "total_time": 0.0, "times": []}
            
            self.metrics[name]["count"] += 1
            self.metrics[name]["total_time"] += elapsed
            self.metrics[name]["times"].append(elapsed)
        
        return elapsed
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB.
        
        Returns:
            Memory usage in megabytes
        """
        try:
            return self.process.memory_info().rss / (1024 * 1024)
        except Exception as e:
            logger.error(f"Error getting memory usage: {str(e)}")
            return 0.0
    
    def get_memory_increase(self) -> float:
        """Get the increase in memory usage since monitoring started.
        
        Returns:
            Memory increase in megabytes
        """
        current = self.get_memory_usage()
        return current - self.baseline_memory
    
    def get_max_memory_usage(self) -> float:
        """Get the maximum memory usage observed.
        
        Returns:
            Maximum memory usage in megabytes
        """
        return self.max_memory
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage for this process.
        
        Returns:
            CPU usage percentage (0-100)
        """
        try:
            return self.process.cpu_percent(interval=0.1)
        except Exception as e:
            logger.error(f"Error getting CPU usage: {str(e)}")
            return 0.0
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        with self.lock:
            metrics_copy = self.metrics.copy()
        
        # Calculate statistics for each operation
        for name, data in metrics_copy.items():
            if data["count"] > 0:
                times = data["times"]
                data["avg_time"] = data["total_time"] / data["count"]
                data["min_time"] = min(times) if times else 0
                data["max_time"] = max(times) if times else 0
                if len(times) > 1:
                    data["std_dev"] = np.std(times)
                else:
                    data["std_dev"] = 0
        
        # Add memory metrics
        metrics_copy["memory"] = {
            "baseline_mb": self.baseline_memory,
            "current_mb": self.get_memory_usage(),
            "max_mb": self.max_memory,
            "increase_mb": self.get_memory_increase()
        }
        
        # Add CPU metrics
        metrics_copy["cpu"] = {
            "current_percent": self.get_cpu_usage()
        }
        
        return metrics_copy
    
    def reset_metrics(self):
        """Reset all collected metrics."""
        with self.lock:
            self.metrics = {}
            self.start_times = {}
            self.baseline_memory = self.get_memory_usage()
            self.max_memory = self.baseline_memory
        
        logger.debug("Performance metrics reset")
    
    def generate_report(self) -> str:
        """Generate a human-readable performance report.
        
        Returns:
            Formatted performance report
        """
        metrics = self.get_metrics()
        
        report = []
        report.append("=" * 80)
        report.append("VSAT Performance Report")
        report.append("=" * 80)
        report.append("")
        
        # Memory section
        report.append("Memory Usage:")
        report.append("-" * 40)
        memory = metrics.get("memory", {})
        report.append(f"Baseline: {memory.get('baseline_mb', 0):.2f} MB")
        report.append(f"Current:  {memory.get('current_mb', 0):.2f} MB")
        report.append(f"Maximum:  {memory.get('max_mb', 0):.2f} MB")
        report.append(f"Increase: {memory.get('increase_mb', 0):.2f} MB")
        report.append("")
        
        # CPU section
        report.append("CPU Usage:")
        report.append("-" * 40)
        cpu = metrics.get("cpu", {})
        report.append(f"Current: {cpu.get('current_percent', 0):.2f}%")
        report.append("")
        
        # Operations timing section
        report.append("Operation Timing:")
        report.append("-" * 40)
        report.append(f"{'Operation':<30} {'Count':>8} {'Avg (s)':>10} {'Min (s)':>10} {'Max (s)':>10} {'Total (s)':>10}")
        report.append("-" * 80)
        
        # Sort operations by total time (descending)
        operations = sorted(
            [(k, v) for k, v in metrics.items() if k not in ("memory", "cpu")],
            key=lambda x: x[1].get("total_time", 0),
            reverse=True
        )
        
        for name, data in operations:
            count = data.get("count", 0)
            avg_time = data.get("avg_time", 0)
            min_time = data.get("min_time", 0)
            max_time = data.get("max_time", 0)
            total_time = data.get("total_time", 0)
            
            report.append(f"{name:<30} {count:>8} {avg_time:>10.3f} {min_time:>10.3f} {max_time:>10.3f} {total_time:>10.3f}")
        
        return "\n".join(report)


# Global performance monitor instance
_performance_monitor = PerformanceMonitor()

def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance.
    
    Returns:
        Global PerformanceMonitor instance
    """
    return _performance_monitor


def timed(operation_name: Optional[str] = None):
    """Decorator to time a function and record metrics.
    
    Args:
        operation_name: Name of the operation (defaults to function name)
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = operation_name or func.__name__
            monitor = get_performance_monitor()
            monitor.start_timer(name)
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = monitor.stop_timer(name)
                logger.debug(f"{name} completed in {elapsed:.3f} seconds")
        return wrapper
    return decorator


class ResourceOptimizer:
    """Optimize resource usage during processing."""
    
    def __init__(self):
        """Initialize the resource optimizer."""
        self.monitor = get_performance_monitor()
        self.memory_threshold_mb = 1024  # 1GB default threshold
        self.cpu_threshold_percent = 80.0
        self.optimization_active = False
    
    def set_memory_threshold(self, threshold_mb: float):
        """Set the memory threshold for optimization.
        
        Args:
            threshold_mb: Memory threshold in megabytes
        """
        self.memory_threshold_mb = threshold_mb
    
    def set_cpu_threshold(self, threshold_percent: float):
        """Set the CPU threshold for optimization.
        
        Args:
            threshold_percent: CPU threshold percentage (0-100)
        """
        self.cpu_threshold_percent = threshold_percent
    
    def start_optimization(self):
        """Start resource optimization."""
        self.optimization_active = True
        self.monitor.start_monitoring()
        logger.info("Resource optimization started")
    
    def stop_optimization(self):
        """Stop resource optimization."""
        self.optimization_active = False
        self.monitor.stop_monitoring()
        logger.info("Resource optimization stopped")
    
    def should_optimize(self) -> bool:
        """Check if optimization should be applied based on current resource usage.
        
        Returns:
            True if optimization should be applied, False otherwise
        """
        if not self.optimization_active:
            return False
        
        current_memory = self.monitor.get_memory_usage()
        current_cpu = self.monitor.get_cpu_usage()
        
        memory_pressure = current_memory >= self.memory_threshold_mb
        cpu_pressure = current_cpu >= self.cpu_threshold_percent
        
        return memory_pressure or cpu_pressure
    
    def get_optimal_chunk_size(self, file_size_mb: float) -> float:
        """Calculate the optimal chunk size for processing based on file size and available resources.
        
        Args:
            file_size_mb: Size of the file in megabytes
            
        Returns:
            Optimal chunk size in seconds
        """
        # Get available memory
        available_memory = psutil.virtual_memory().available / (1024 * 1024)
        
        # Base chunk size on available memory (aim to use at most 25% of available memory)
        memory_based_chunk_mb = available_memory * 0.25
        
        # Convert to seconds based on approximate audio size (16-bit stereo at 44.1kHz)
        # ~10MB per minute of audio
        memory_based_chunk_seconds = (memory_based_chunk_mb / 10.0) * 60.0
        
        # Ensure chunk size is reasonable (between 5 and 60 seconds)
        chunk_seconds = max(5.0, min(60.0, memory_based_chunk_seconds))
        
        # If under memory pressure, reduce chunk size further
        if self.should_optimize():
            chunk_seconds *= 0.5
        
        logger.debug(f"Calculated optimal chunk size: {chunk_seconds:.2f} seconds")
        return chunk_seconds
    
    def optimize_numpy_operations(self, array_size_threshold_mb: float = 100.0):
        """Configure NumPy for optimal performance based on current resources.
        
        Args:
            array_size_threshold_mb: Threshold for large array operations in MB
        """
        # If under resource pressure, optimize for memory usage
        if self.should_optimize():
            # Use more conservative settings for large array operations
            np.seterr(over='warn', divide='warn', invalid='warn')
            logger.debug("NumPy configured for memory conservation")
        else:
            # Use default settings
            np.seterr(all='warn')
            logger.debug("NumPy configured for default operation")


# Global resource optimizer instance
_resource_optimizer = ResourceOptimizer()

def get_resource_optimizer() -> ResourceOptimizer:
    """Get the global resource optimizer instance.
    
    Returns:
        Global ResourceOptimizer instance
    """
    return _resource_optimizer


class BatchProcessingOptimizer:
    """Optimize batch processing operations."""
    
    def __init__(self):
        """Initialize the batch processing optimizer."""
        self.monitor = get_performance_monitor()
        self.optimizer = get_resource_optimizer()
        self.max_concurrent_tasks = self._determine_optimal_concurrency()
    
    def _determine_optimal_concurrency(self) -> int:
        """Determine the optimal number of concurrent tasks based on system resources.
        
        Returns:
            Optimal number of concurrent tasks
        """
        # Get CPU count, but leave at least one core free for system
        cpu_count = max(1, psutil.cpu_count(logical=False) - 1)
        
        # Consider memory constraints
        total_memory_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)
        
        # Heuristic: 1 task per 4GB of RAM, but at least 1 and at most CPU count
        memory_based_tasks = max(1, int(total_memory_gb / 4))
        
        # Take the minimum of CPU-based and memory-based task counts
        optimal_tasks = min(cpu_count, memory_based_tasks)
        
        logger.debug(f"Determined optimal concurrency: {optimal_tasks} tasks")
        return optimal_tasks
    
    def get_max_concurrent_tasks(self) -> int:
        """Get the maximum number of concurrent tasks to run.
        
        Returns:
            Maximum number of concurrent tasks
        """
        # Reduce concurrency if under resource pressure
        if self.optimizer.should_optimize():
            return max(1, self.max_concurrent_tasks // 2)
        
        return self.max_concurrent_tasks
    
    def optimize_batch_order(self, files: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Optimize the processing order of a batch of files.
        
        Args:
            files: List of (file_path, file_size_mb) tuples
            
        Returns:
            Optimized list of files for processing
        """
        # Process smaller files first to provide quick feedback
        sorted_files = sorted(files, key=lambda x: x[1])
        
        # If we have many files, interleave some larger files to balance resource usage
        if len(sorted_files) > 10:
            result = []
            small_files = sorted_files[:len(sorted_files)//2]
            large_files = sorted_files[len(sorted_files)//2:]
            
            # Take 2 small files, then 1 large file
            small_idx = 0
            large_idx = 0
            
            while small_idx < len(small_files) or large_idx < len(large_files):
                # Add up to 2 small files
                for _ in range(2):
                    if small_idx < len(small_files):
                        result.append(small_files[small_idx])
                        small_idx += 1
                
                # Add 1 large file
                if large_idx < len(large_files):
                    result.append(large_files[large_idx])
                    large_idx += 1
            
            return result
        
        return sorted_files


# Global batch processing optimizer instance
_batch_optimizer = BatchProcessingOptimizer()

def get_batch_optimizer() -> BatchProcessingOptimizer:
    """Get the global batch processing optimizer instance.
    
    Returns:
        Global BatchProcessingOptimizer instance
    """
    return _batch_optimizer 