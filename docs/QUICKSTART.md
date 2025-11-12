# Yale Jobs Quick Start

**TL;DR: HuggingFace Jobs for Yale HPC**

## Install

```bash
pip install -e .
```

## Configure

Create `config.yaml`:

```yaml
alias: grace.hpc.yale.edu
env: my-conda-env
job_dir: project/jobs
result_dir: project/results
2fa: true
```

## Run OCR (3 ways)

### 1. CLI - Simplest

```bash
yale jobs ocr pdfs/ output --source-type pdf --gpus v100:2
```

### 2. Python Function

```python
from yale import run_ocr_job

job = run_ocr_job(
    data_source="pdfs/",
    output_dataset="output",
    source_type="pdf",
    gpus="v100:2"
)
```

### 3. Full Control

```python
from yale import YaleJobs

with YaleJobs() as yale:
    yale.connect()
    job = yale.submit_job(
        script=my_script,
        data_source="pdfs/",
        gpus="v100:2"
    )
```

## Data Sources

```python
# PDFs
source_type="pdf"
data_source="path/to/pdfs/"

# IIIF Manifest
source_type="iiif"
data_source="https://example.com/manifest.json"

# Image Directory
source_type="directory"
data_source="path/to/images/"

# Web URLs
source_type="web"
data_source="urls.txt"  # one URL per line

# HuggingFace Dataset
source_type="hf"
data_source="username/dataset-name"
```

## Job Management

```bash
# Check status
yale jobs status 12345

# View logs
yale jobs logs --job-name my-job

# Download results
yale jobs download --job-name my-job --output-dir ./results

# Cancel
yale jobs cancel 12345
```

## Common Options

```bash
--gpus p100:2          # GPU specification
--batch-size 32        # Batch size
--max-samples 100      # Limit for testing
--wait                 # Wait for completion
--job-name my-job      # Custom job name
```

## GPU Types

- `p100:1` or `p100:2` - Standard
- `v100:1` or `v100:2` - Better
- `a100:1` or `a100:2` - Best

## Examples

```bash
# Test with 10 samples
yale jobs ocr pdfs/ test --source-type pdf --max-samples 10

# Production run
yale jobs ocr pdfs/ output --source-type pdf --gpus v100:2 --batch-size 32

# IIIF collection
yale jobs ocr https://example.com/manifest.json output --source-type iiif

# Wait for completion and download
yale jobs ocr pdfs/ output --source-type pdf --wait
yale jobs download --job-name yale-ocr --output-dir ./results
```

## Python Examples

```python
# Simple OCR
from yale import run_ocr_job

job = run_ocr_job("pdfs/", "output", source_type="pdf")
print(f"Job ID: {job.job_id}")

# Custom script
from yale import run_job

script = """
from datasets import load_from_disk
dataset = load_from_disk("dataset")
print(f"Processing {len(dataset)} items")
"""

job = run_job(script, data_source="data/", gpus="v100:2")

# Check status
status = job.get_status()
print(status['state'])

# Download results
job.download_results("./results")
```

## Troubleshooting

**Connection fails?**
```bash
ssh netid@grace.hpc.yale.edu  # Test manually
```

**Job fails?**
```bash
yale jobs logs --job-name my-job
```

**Out of memory?**
- Reduce `--batch-size`
- Add `--memory 64G`

**Full documentation:** See [README.md](README.md) and [USAGE.md](USAGE.md)

