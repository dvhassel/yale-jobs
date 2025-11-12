# UV Integration

Yale Jobs now uses [UV](https://docs.astral.sh/uv/) for automatic dependency management using PEP 723 inline script metadata.

## What Changed

### Before (Manual Dependencies)
```bash
# Had to manually install packages
pip install datasets vllm torch pillow
python my_script.py
```

### After (UV Automatic)
```bash
# UV installs dependencies automatically!
uv run my_script.py
```

## How It Works

### 1. PEP 723 Metadata in Scripts

All generated Python scripts now include inline dependency metadata:

```python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets",
#     "huggingface-hub[hf_transfer]",
#     "pillow",
#     "vllm>=0.9.1",
#     "tqdm",
#     "toolz",
#     "torch",
# ]
# ///

import datasets
from vllm import LLM
# ... rest of script
```

### 2. Automatic UV Installation

The SLURM batch script automatically installs UV if needed:

```bash
#!/bin/bash
#SBATCH --job-name=my-job
#SBATCH --gpus=h200:1

# Install uv for dependency management
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Activate conda environment for system packages (CUDA, etc.)
source ~/.bashrc
conda activate vllm

# Run scripts with uv
uv run preprocessing.py && uv run ocr.py
```

### 3. Dependency Resolution

UV automatically:
- Reads the PEP 723 metadata
- Creates an isolated environment
- Installs exact versions
- Caches packages for fast reuse
- Uses Conda's CUDA/system packages

## Benefits

### ✅ Zero Manual Setup
- No need to manually install packages
- No environment.yml or requirements.txt files
- Dependencies declared right in the script

### ✅ Reproducibility
- Each script declares exact dependencies
- Version pinning built-in
- Isolated environments prevent conflicts

### ✅ Speed
- UV is 10-100x faster than pip
- Aggressive caching
- Parallel downloads

### ✅ Simplicity
- One tool (`uv run`) does everything
- Works with existing Conda CUDA packages
- No virtual environment management

## Dependency Types

### Python Packages (UV)
```python
# /// script
# dependencies = [
#     "datasets",
#     "vllm>=0.9.1",
#     "torch",
# ]
# ///
```
UV handles: datasets, vllm, torch, pillow, etc.

### System Packages (Conda)
```yaml
# config.yaml
env: vllm  # Conda env with CUDA
```
Conda provides: CUDA, cuDNN, system libraries

**Best of both worlds**: UV for Python, Conda for CUDA/system!

## Generated Scripts

### Preprocessing Script (PDF/IIIF)
```python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets",
#     "huggingface-hub",
#     "pillow",
#     "requests",
#     "pypdfium2",
# ]
# ///

# Converts PDFs/IIIF to HuggingFace datasets
# Runs on cluster with UV
```

### OCR Script (DoTS.ocr)
```python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets",
#     "huggingface-hub[hf_transfer]",
#     "pillow",
#     "vllm>=0.9.1",
#     "tqdm",
#     "toolz",
#     "torch",
# ]
# ///

# Runs DoTS.ocr with vLLM
# Uses file:// URLs for efficient image loading
```

## Example: Full Job Flow

```bash
# Submit job
yale jobs ocr manifests.txt output \
    --hpc-process \
    --gpus h200:1 \
    --partition gpu_h200

# What happens on the cluster:
# 1. SLURM starts job
# 2. Installs UV (if needed)
# 3. Activates Conda (for CUDA)
# 4. uv run preprocessing.py
#    - UV reads PEP 723 metadata
#    - Installs: datasets, pillow, requests, pypdfium2
#    - Downloads IIIF images
#    - Creates dataset
# 5. uv run ocr.py
#    - UV reads PEP 723 metadata
#    - Installs: vllm, torch, tqdm, toolz
#    - Runs DoTS.ocr
#    - Saves results
```

## Comparison

| Feature | Old (pip/conda) | New (UV) |
|---------|----------------|----------|
| Install deps | Manual | Automatic |
| Env management | Manual | Automatic |
| Reproducibility | Medium | High |
| Speed | Slow | Fast (10-100x) |
| Caching | Basic | Aggressive |
| Isolation | Virtual envs | Per-script |
| Setup time | Minutes | Seconds |

## Migration Guide

If you have existing scripts, add PEP 723 metadata:

```python
# Before
import datasets
from vllm import LLM

# After
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets",
#     "vllm>=0.9.1",
# ]
# ///

import datasets
from vllm import LLM
```

Then run with `uv run` instead of `python`:
```bash
# Before
python my_script.py

# After
uv run my_script.py
```

## Resources

- [UV Documentation](https://docs.astral.sh/uv/)
- [PEP 723 Specification](https://peps.python.org/pep-0723/)
- [Yale Jobs Examples](EXAMPLES.md)


