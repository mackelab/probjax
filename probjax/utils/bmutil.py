import os
import threading
import time

import psutil
from pynvml import (
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetUtilizationRates,
    nvmlInit,
    nvmlShutdown,
)


def benchmark(
    func,
    *args,
    max_time=5,
    track_gpu=True,
    track_cpu=True,
    track_mem=True,
    track_disk=False,
    **kwargs,
):
    """Benchmark the time taken by a function to execute, and return the result of the function."""
    result = func(*args, **kwargs)  # Pre-run to ensure that the function is compiled
    start = time.time()
    count = 0
    # Run the function such that the time takes around 5 seconds
    # Average over multiple runs
    trackers = []
    if track_gpu:
        gpu_tracker = GPUUtilizationTracker()
        trackers.append(gpu_tracker)
    if track_cpu:
        cpu_tracker = CPUUtilizationTracker()
        trackers.append(cpu_tracker)
    if track_mem:
        mem_tracker = MemoryUtilizationTracker()
        trackers.append(mem_tracker)
    if track_disk:
        disk_tracker = DiskUtilizationTracker()
        trackers.append(disk_tracker)

    for tracker in trackers:
        tracker.start()

    while time.time() - start < max_time:
        _ = func(*args, **kwargs)
        count += 1

    for tracker in trackers:
        tracker.stop()

    end_time = time.time()
    # Return time in best possible units
    time_taken = (end_time - start) / count
    if time_taken < 1e-6:
        time_taken *= 1e9
        unit = "ns"
    elif time_taken < 1e-3:
        time_taken *= 1e6
        unit = "us"
    elif time_taken < 1:
        time_taken *= 1e3
        unit = "ms"
    else:
        unit = "s"
    print(f"Average time taken: {time_taken:.2f} {unit}")
    return result


class OnlineMeanStdEstimator:
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, new_value):
        self.count += 1
        delta = new_value - self.mean
        self.mean += delta / self.count
        delta2 = new_value - self.mean
        self.M2 += delta * delta2

    def get_mean(self):
        return self.mean

    def get_std(self):
        if self.count < 2:
            return 0.0
        return (self.M2 / (self.count - 1)) ** 0.5


class Tracker:
    running = False

    def __enter__(self):
        self.running = False
        self.start()
        return self

    def _track_quantity(self):
        raise NotImplementedError("This method should be implemented by the subclass")

    def get_summary(self):
        raise NotImplementedError("This method should be implemented by the subclass")

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._track_quantity)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def __repr__(self) -> str:
        return self.get_summary()


class GPUUtilizationTracker(Tracker):
    def __init__(self, device_idx=0, verbose=False):
        self.device_idx = device_idx
        self.gpu_utilization = OnlineMeanStdEstimator()
        self.memory_utilization = OnlineMeanStdEstimator()
        self.verbose = verbose

    def _track_quantity(self):
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(self.device_idx)
        while self.running:
            utilization = nvmlDeviceGetUtilizationRates(handle)
            self.gpu_utilization.update(utilization.gpu)
            self.memory_utilization.update(utilization.memory)
            if self.verbose:
                print(self.get_summary(), end="\r")
            time.sleep(0.01)
        print(self.get_summary())
        nvmlShutdown()

    def get_summary(self):
        return f"GPU Utilization: {int(self.gpu_utilization.get_mean())}% +/- {int(self.gpu_utilization.get_std())}%, GPU Memory Utilization: {int(self.memory_utilization.get_mean())}% +/- {int(self.memory_utilization.get_std())}% "


class CPUUtilizationTracker(Tracker):
    def __init__(self, pid=None, verbose=False):
        if pid is None:
            pid = os.getpid()
        self.pid = pid
        self.cpu_utilization = OnlineMeanStdEstimator()
        self.running = False
        self.verbose = verbose

    def _track_quantity(self):
        process = psutil.Process(self.pid)
        cpu_count = psutil.cpu_count()
        while self.running:
            cpu_utilization = process.cpu_percent() / cpu_count
            self.cpu_utilization.update(cpu_utilization)
            time.sleep(0.01)
            if self.verbose:
                print(self.get_summary(), end="\r")
        print(self.get_summary())

    def get_summary(self):
        return f"CPU Utilization: {int(self.cpu_utilization.get_mean())}% +/- {int(self.cpu_utilization.get_std())}%"


class MemoryUtilizationTracker(Tracker):
    def __init__(self, pid=None, verbose=False):
        if pid is None:
            pid = os.getpid()
        self.pid = pid
        self.memory_utilization = OnlineMeanStdEstimator()
        self.running = False
        self.verbose = verbose

    def _track_quantity(self):
        process = psutil.Process(self.pid)
        while self.running:
            memory_utilization = process.memory_percent()
            self.memory_utilization.update(memory_utilization)
            time.sleep(0.01)
            if self.verbose:
                print(self.get_summary(), end="\r")
        print(self.get_summary())

    def get_summary(self):
        return f"Memory Utilization: {int(self.memory_utilization.get_mean())}% +/- {int(self.memory_utilization.get_std())}%"


class DiskUtilizationTracker(Tracker):
    def __init__(self, path=None, verbose=False):
        if path is None:
            path = os.getcwd()
        self.path = path
        self.disk_utilization = OnlineMeanStdEstimator()
        self.running = False
        self.verbose = verbose

    def _track_quantity(self):
        while self.running:
            disk_utilization = psutil.disk_usage(self.path).percent
            self.disk_utilization.update(disk_utilization)
            time.sleep(0.01)
            if self.verbose:
                print(self.get_summary(), end="\r")
        print(self.get_summary())

    def get_summary(self):
        return f"Disk Utilization: {int(self.disk_utilization.get_mean())}% +/- {int(self.disk_utilization.get_std())}%"
