from __future__ import annotations

import os
import platform
from typing import Any

from .contracts import (
    CapabilityMatrix,
    CpuProfile,
    DiskProfile,
    GpuProfile,
    HardwareSnapshot,
    HardwareTier,
    MemoryProfile,
    ProfilerError,
)

BYTES_PER_GIB = 1024 ** 3


class ProbeSource:
    def cpu(self) -> CpuProfile: ...
    def memory(self) -> MemoryProfile: ...
    def gpus(self) -> tuple[GpuProfile, ...]: ...
    def disks(self) -> tuple[DiskProfile, ...]: ...


class LiveProbeSource(ProbeSource):
    def cpu(self) -> CpuProfile:
        try:
            import psutil
        except ImportError as exc:
            raise ProbeUnavailableError("psutil not installed") from exc
        frequency = psutil.cpu_freq()
        return CpuProfile(
            cores_physical=psutil.cpu_count(logical=False) or 1,
            cores_logical=psutil.cpu_count(logical=True) or 1,
            frequency_mhz=frequency.current if frequency else None,
            architecture=platform.machine(),
        )

    def memory(self) -> MemoryProfile:
        try:
            import psutil
        except ImportError as exc:
            raise ProbeUnavailableError("psutil not installed") from exc
        virtual = psutil.virtual_memory()
        return MemoryProfile(total_bytes=virtual.total, available_bytes=virtual.available)

    def gpus(self) -> tuple[GpuProfile, ...]:
        detected: list[GpuProfile] = []
        try:
            import pynvml

            pynvml.nvmlInit()
            for index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8")
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                detected.append(GpuProfile(
                    name=name, vram_bytes=memory_info.total, supports_cuda=True,
                ))
        except Exception:
            pass
        if platform.system() == "Darwin" and not detected:
            detected.append(GpuProfile(name="Apple Silicon GPU", supports_metal=True))
        return tuple(detected)

    def disks(self) -> tuple[DiskProfile, ...]:
        try:
            import psutil
        except ImportError as exc:
            raise ProbeUnavailableError("psutil not installed") from exc
        partitions: list[DiskProfile] = []
        for partition in psutil.disk_partitions(all=False)[:4]:
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                partitions.append(DiskProfile(
                    total_bytes=usage.total, free_bytes=usage.free, is_ssd=None,
                ))
            except (PermissionError, OSError):
                continue
        return tuple(partitions)


class StaticProbeSource(ProbeSource):
    def __init__(self, snapshot: HardwareSnapshot) -> None:
        self._snapshot = snapshot

    def cpu(self) -> CpuProfile:
        return self._snapshot.cpu

    def memory(self) -> MemoryProfile:
        return self._snapshot.memory

    def gpus(self) -> tuple[GpuProfile, ...]:
        return self._snapshot.gpus

    def disks(self) -> tuple[DiskProfile, ...]:
        return self._snapshot.disks


def classify_tier(snapshot: HardwareSnapshot) -> HardwareTier:
    score = 0.0
    score += min(snapshot.cpu.cores_physical, 16) * 0.6
    ram_gib = snapshot.memory.total_bytes / BYTES_PER_GIB
    score += min(ram_gib, 64) * 0.5
    if snapshot.primary_gpu and snapshot.primary_gpu.has_acceleration:
        vram_gib = snapshot.primary_gpu.vram_bytes / BYTES_PER_GIB
        score += min(vram_gib, 24) * 0.8
    if score >= 40:
        return HardwareTier.SERVER
    if score >= 22:
        return HardwareTier.WORKSTATION
    if score >= 12:
        return HardwareTier.DESKTOP
    if score >= 6:
        return HardwareTier.LAPTOP
    return HardwareTier.MOBILE


def build_capability_matrix(snapshot: HardwareSnapshot) -> CapabilityMatrix:
    tier = classify_tier(snapshot)
    vram_gib = snapshot.total_vram_bytes / BYTES_PER_GIB
    ram_gib = snapshot.memory.total_bytes / BYTES_PER_GIB

    can_llm = vram_gib >= 4 or ram_gib >= 16
    if vram_gib >= 48:
        max_params = 70.0
    elif vram_gib >= 24:
        max_params = 34.0
    elif vram_gib >= 8:
        max_params = 8.0
    elif ram_gib >= 16:
        max_params = 3.0
    else:
        max_params = 1.5

    notes: list[str] = []
    if not snapshot.primary_gpu or not snapshot.primary_gpu.has_acceleration:
        notes.append("no hardware acceleration detected; CPU-only inference")
    if snapshot.primary_disk and snapshot.primary_disk.free_ratio < 0.15:
        notes.append("low disk headroom; large model downloads may fail")

    return CapabilityMatrix(
        tier=tier,
        can_run_local_llm=can_llm,
        recommended_max_params_billion=max_params,
        supports_fine_tuning=vram_gib >= 16,
        supports_parallel_pipelines=snapshot.cpu.cores_logical >= 8,
        notes=tuple(notes),
    )


def collect_snapshot(source: ProbeSource) -> HardwareSnapshot:
    return HardwareSnapshot(
        cpu=source.cpu(),
        memory=source.memory(),
        gpus=source.gpus(),
        disks=source.disks(),
        platform=platform.platform(),
        python_version=platform.python_version(),
    )
