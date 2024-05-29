import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from hw_profiler import (
    CapabilityMatrix,
    CpuProfile,
    DiskProfile,
    GpuProfile,
    HardwareSnapshot,
    HardwareTier,
    MemoryProfile,
    ProfilerError,
    build_capability_matrix,
    classify_tier,
)

GiB = 1024 ** 3


def make_snapshot(
    cores: int = 4,
    ram_gib: float = 8.0,
    vram_gib: float = 0.0,
    free_disk_ratio: float = 0.5,
) -> HardwareSnapshot:
    gpu = (
        GpuProfile(name="RTX", vram_bytes=int(vram_gib * GiB), supports_cuda=True),
    ) if vram_gib else ()
    return HardwareSnapshot(
        cpu=CpuProfile(cores_physical=cores, cores_logical=cores, frequency_mhz=None, architecture="x86_64"),
        memory=MemoryProfile(total_bytes=int(ram_gib * GiB), available_bytes=int(ram_gib * GiB * 0.5)),
        gpus=gpu,
        disks=(DiskProfile(total_bytes=500 * GiB, free_bytes=int(500 * GiB * free_disk_ratio)),),
    )


def test_cpu_validation_rejects_illogical_cores():
    with pytest.raises(ProfilerError):
        CpuProfile(cores_physical=8, cores_logical=4, frequency_mhz=None, architecture="x86")


def test_memory_validation_rejects_negative_free():
    with pytest.raises(ProfilerError):
        MemoryProfile(total_bytes=100, available_bytes=200)


def test_tier_classification_low_end_mobile():
    assert classify_tier(make_snapshot(cores=2, ram_gib=3)) == HardwareTier.MOBILE


def test_tier_classification_mid_laptop():
    tier = classify_tier(make_snapshot(cores=4, ram_gib=16))
    assert tier in {HardwareTier.LAPTOP, HardwareTier.DESKTOP}


def test_tier_classification_server_with_big_gpu():
    snapshot = make_snapshot(cores=32, ram_gib=128, vram_gib=80)
    assert classify_tier(snapshot) == HardwareTier.SERVER


def test_primary_gpu_picked_by_vram():
    snapshot = HardwareSnapshot(
        cpu=CpuProfile(2, 2, None, "arm"),
        memory=MemoryProfile(8 * GiB, 4 * GiB),
        gpus=(
            GpuProfile("small", vram_bytes=2 * GiB, supports_cuda=True),
            GpuProfile("big", vram_bytes=24 * GiB, supports_cuda=True),
        ),
    )
    assert snapshot.primary_gpu.name == "big"
    assert snapshot.total_vram_bytes == 26 * GiB


def test_llm_capability_thresholds():
    no_accel = build_capability_matrix(make_snapshot(cores=4, ram_gib=8))
    assert not no_accel.can_run_local_llm
    mid_gpu = build_capability_matrix(make_snapshot(cores=8, ram_gib=32, vram_gib=12))
    assert mid_gpu.can_run_local_llm


def test_param_recommendation_scales_with_vram():
    small = build_capability_matrix(make_snapshot(vram_gib=6)).recommended_max_params_billion
    large = build_capability_matrix(make_snapshot(vram_gib=64)).recommended_max_params_billion
    assert small < large
    assert large == 70.0


def test_finetuning_requires_vram_headroom():
    assert not build_capability_matrix(make_snapshot(vram_gib=4)).supports_fine_tuning
    assert build_capability_matrix(make_snapshot(vram_gib=24)).supports_fine_tuning


def test_low_disk_note_emitted():
    matrix = build_capability_matrix(make_snapshot(free_disk_ratio=0.05))
    assert any("disk" in note for note in matrix.notes)


def test_no_acceleration_note():
    matrix = build_capability_matrix(make_snapshot())
    assert any("CPU-only" in note for note in matrix.notes)
