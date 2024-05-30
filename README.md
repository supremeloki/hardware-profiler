# hw-profiler

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Detects the machine it runs on — CPU, RAM, GPUs, disks — classifies it into a hardware tier, and answers the practical question: *what can this box actually run?*

## 🚀 Overview

Adaptive software needs to know its hardware. `hw-profiler` probes live system info (via optional `psutil`/`pynvml`) or accepts a static snapshot, validates physical consistency (logical cores can't exceed physical, available can't exceed total), scores the machine into a tier from **MOBILE** to **SERVER**, and emits a `CapabilityMatrix` — local-LLM feasibility, recommended max model params, fine-tuning eligibility, and human-readable notes.

## ✨ Features

- **Frozen contracts with self-validation:** impossible hardware raises at construction
- **Pluggable probe sources:** `LiveProbeSource` (real machine) or `StaticProbeSource` (tests/CI) behind one `ProbeSource` protocol
- **Tier classification:** weighted score across cores/RAM/VRAM → MOBILE · LAPTOP · DESKTOP · WORKSTATION · SERVER
- **LLM sizing:** recommended max parameter count scales with VRAM (1.5B → 70B)
- **Fine-tuning gate:** requires ≥16GB VRAM headroom
- **Actionable notes:** "no acceleration; CPU-only inference", "low disk headroom", …
- **Graceful degradation:** missing NVML/CUDA simply yields fewer GPUs, never a crash
- **Zero hard dependencies** (psutil/pynvml optional)

## 🚧 Structure

```
hardware-profiler/
├── src/hw_profiler/
│   ├── __init__.py
│   ├── contracts.py
│   └── core.py
├── tests/
│   └── test_core.py
├── README.md
└── pyproject.toml
```

## 📦 Installation

```bash
git clone https://github.com/supremeloki/hardware-profiler.git
cd hardware-profiler
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## 📋 Requirements

- Python 3.11+
- Optional: `psutil`, `pynvml` for live probing

## 🏃 Quick Start

```python
from hw_profiler import LiveProbeSource, StaticProbeSource, collect_snapshot, build_capability_matrix

snapshot = collect_snapshot(LiveProbeSource())
matrix = build_capability_matrix(snapshot)

print(matrix.tier)
print(matrix.recommended_max_params_billion)
print(matrix.supports_fine_tuning)
for note in matrix.notes:
    print(note)
```

### Deterministic tests

```python
from pathlib import Path
from hw_profiler import StaticProbeSource

fake = HardwareSnapshot(cpu=..., memory=..., gpus=(...), disks=(...))
source = StaticProbeSource(fake)
```

## 🔧 Error Handling

```text
ProfilerError            # inconsistent profile data
└── ProbeUnavailableError  # psutil/pynvml absent for live probing
```

## 🧪 Testing

```bash
pytest tests/ -v
```

## 📝 Code Quality

- Full type hints (`X | None` style), frozen dataclasses throughout
- Zero comments — names carry the meaning
- `ruff` clean

## 📄 License

MIT — see [LICENSE](LICENSE).

## 👤 Author

**Kooroush Masoumi**

---

⭐ Star this repo if you find it useful!
