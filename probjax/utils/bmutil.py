import os
import threading
import time
from contextlib import suppress

import jax
import psutil
from pynvml import (
    NVMLError,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetUtilizationRates,
    nvmlInit,
    nvmlShutdown,
)


class Benchmark:
    """Context manager to benchmark a code block while collecting resource metrics."""

    def __init__(
        self,
        *,
        track_gpu=True,
        track_cpu=True,
        track_mem=True,
        track_disk=False,
    ):
        self.track_gpu = track_gpu
        self.track_cpu = track_cpu
        self.track_mem = track_mem
        self.track_disk = track_disk

        self.elapsed = None

        self._trackers = []
        self._trackers_running = False
        self._start_time = None
        self._end_time = None

    def __enter__(self):
        self._prepare_trackers()
        self._start_trackers()
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end_time = time.time()
        self.elapsed = (
            self._end_time - self._start_time if self._start_time is not None else None
        )
        # Ensure trackers stop even when errors occur inside the context.
        self._stop_trackers()
        self._report_elapsed_time()
        return False

    def _report_elapsed_time(self):
        if self.elapsed is None:
            print("Benchmark elapsed time unavailable.")
            return

        time_taken = self.elapsed
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
        print(f"Elapsed time: {time_taken:.2f} {unit}")

    def _prepare_trackers(self):
        self._trackers = []
        if self.track_gpu and self._gpu_is_available():
            self._trackers.append(GPUUtilizationTracker())
        elif self.track_gpu:
            self.track_gpu = False

        if self.track_cpu:
            self._trackers.append(CPUUtilizationTracker())
        if self.track_mem:
            self._trackers.append(MemoryUtilizationTracker())
        if self.track_disk:
            self._trackers.append(DiskUtilizationTracker())

    def _start_trackers(self):
        if self._trackers_running:
            return
        for tracker in self._trackers:
            tracker.start()
        self._trackers_running = True

    def _stop_trackers(self):
        if not self._trackers_running:
            return
        for tracker in self._trackers:
            tracker.stop()
        self._trackers_running = False

    @staticmethod
    def _gpu_is_available():
        try:
            devices = jax.devices()
        except RuntimeError:
            return False
        if not any(device.platform == "gpu" for device in devices):
            return False
        try:
            nvmlInit()
        except NVMLError:
            return False
        finally:
            with suppress(NVMLError):
                nvmlShutdown()
        return True


def benchmark(
    *,
    track_gpu=True,
    track_cpu=True,
    track_mem=True,
    track_disk=False,
):
    """Create a Benchmark context manager."""
    return Benchmark(
        track_gpu=track_gpu,
        track_cpu=track_cpu,
        track_mem=track_mem,
        track_disk=track_disk,
    )


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
        return (
            f"GPU Utilization: {int(self.gpu_utilization.get_mean())}% "
            f"+/- {int(self.gpu_utilization.get_std())}%,"
            f"GPU Memory Utilization: {int(self.memory_utilization.get_mean())}% "
            f"+/- {int(self.memory_utilization.get_std())}% "
        )


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
        cpu_count = cpu_count if cpu_count is not None else 1
        while self.running:
            cpu_utilization = process.cpu_percent() / cpu_count
            self.cpu_utilization.update(cpu_utilization)
            time.sleep(0.01)
            if self.verbose:
                print(self.get_summary(), end="\r")
        print(self.get_summary())

    def get_summary(self):
        return (
            f"CPU Utilization: {int(self.cpu_utilization.get_mean())}% "
            f"+/- {int(self.cpu_utilization.get_std())}%"
        )


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
        return (
            f"Memory Utilization: {int(self.memory_utilization.get_mean())}% "
            f"+/- {int(self.memory_utilization.get_std())}%"
        )


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
        return (
            f"Disk Utilization: {int(self.disk_utilization.get_mean())}% +/-"
            f"{int(self.disk_utilization.get_std())}%"
        )
