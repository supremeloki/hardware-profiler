from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class HardwareTier(Enum):
    MOBILE = "mobile"
    LAPTOP = "laptop"
    DESKTOP = "desktop"
    WORKSTATION = "workstation"
    SERVER = "server"


class ProfilerError(Exception):
    pass


class ProbeUnavailableError(ProfilerError):
    pass


@dataclass(frozen=True)
class CpuProfile:
    cores_physical: int
    cores_logical: int
    frequency_mhz: float | None
    architecture: str

    def __post_init__(self) -> None:
        if self.cores_logical < self.cores_physical:
            raise ProfilerError("logical cores cannot exceed physical cores")


@dataclass(frozen=True)
class MemoryProfile:
    total_bytes: int
    available_bytes: int

    def __post_init__(self) -> None:
        if self.available_bytes > self.total_bytes:
            raise ProfilerError("available memory exceeds total")


@dataclass(frozen=True)
class GpuProfile:
    name: str
    vram_bytes: int = 0
    supports_cuda: bool = False
    supports_metal: bool = False
    supports_directml: bool = False

    @property
    def has_acceleration(self) -> bool:
        return self.supports_cuda or self.supports_metal or self.supports_directml


@dataclass(frozen=True)
class DiskProfile:
    total_bytes: int
    free_bytes: int
    is_ssd: bool | None = None

    @property
    def free_ratio(self) -> float:
        return self.free_bytes / self.total_bytes if self.total_bytes else 0.0


@dataclass(frozen=True)
class HardwareSnapshot:
    cpu: CpuProfile
    memory: MemoryProfile
    gpus: tuple[GpuProfile, ...] = ()
    disks: tuple[DiskProfile, ...] = ()
    platform: str = ""
    python_version: str = ""

    @property
    def primary_gpu(self) -> GpuProfile | None:
        if not self.gpus:
            return None
        return max(self.gpus, key=lambda gpu: gpu.vram_bytes)

    @property
    def total_vram_bytes(self) -> int:
        return sum(gpu.vram_bytes for gpu in self.gpus)

    @property
    def primary_disk(self) -> DiskProfile | None:
        if not self.disks:
            return None
        return max(self.disks, key=lambda disk: disk.free_bytes)


@dataclass(frozen=True)
class CapabilityMatrix:
    tier: HardwareTier
    can_run_local_llm: bool
    recommended_max_params_billion: float
    supports_fine_tuning: bool
    supports_parallel_pipelines: bool
    notes: tuple[str, ...] = field(default_factory=tuple)
