# Yale Jobs

**Job management for Yale's HPC cluster - like HuggingFace Jobs for Yale**

Yale Jobs provides a simple, HuggingFace-style API for running jobs on Yale's HPC cluster. It handles SSH connections (with 2FA), data preparation from multiple sources (PDFs, IIIF, directories, web, HuggingFace datasets), job submission, and monitoring.

## Features

✨ **Simple API** - HuggingFace-style job submission  
🔐 **2FA Support** - Seamless authentication with Yale's cluster  
📁 **Multiple Data Sources** - PDFs, IIIF manifests, directories, web URLs, HF datasets  
🚀 **OCR Ready** - Built-in support for DoTS.ocr and other models  
📊 **Job Monitoring** - Track status and download results  
🐍 **Python SDK & CLI** - Use programmatically or from command line

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/yale-jobs.git
cd yale-jobs

# Install with pip
pip install -e .

# Or install with OCR dependencies
pip install -e ".[ocr]"
```

## Quick Start

### 1. Configure

Create a `config.yaml` file:

```yaml
alias: yale-cluster  # Your cluster hostname
login: true
env: qwen  # Conda environment name
job_dir: proj*/shared/test
result_dir: proj*/shared/test/results
2fa: true
```

### 2. Run OCR Job (CLI)

```bash
# OCR on PDFs
yale jobs ocr path/to/pdfs output-dataset --source-type pdf --gpus v100:2

# OCR on IIIF manifest
yale jobs ocr https://example.com/manifest.json output --source-type iiif

# OCR on image directory
yale jobs ocr path/to/images output --source-type directory --batch-size 32

# Check status
yale jobs status 12345

# Download results
yale jobs download --job-name my-job --output-dir ./results

# View logs
yale jobs logs --job-name my-job
```

### 3. Run OCR Job (Python SDK)

```python
from yale import run_ocr_job

# Run OCR on a directory of PDFs
job = run_ocr_job(
    data_source="manuscripts/",
    output_dataset="manuscripts-ocr",
    source_type="pdf",
    gpus="v100:2",
    batch_size=32
)

print(f"Job ID: {job.job_id}")

# Check status
status = job.get_status()
print(f"State: {status['state']}")

# Download results
job.download_results("./results")
```

## Data Sources

Yale Jobs supports multiple data sources:

### 📄 PDFs

```python
from yale import run_ocr_job

job = run_ocr_job(
    data_source="path/to/pdfs/",  # Single PDF or directory
    output_dataset="pdf-ocr",
    source_type="pdf",
    gpus="v100:2"
)
```

### 🖼️ IIIF Manifests

Supports both IIIF Presentation API v2 and v3:

```python
job = run_ocr_job(
    data_source="https://example.com/iiif/manifest.json",
    output_dataset="iiif-ocr",
    source_type="iiif",
    gpus="p100:2"
)
```

### 🌐 Web URLs

```python
# Single URL
job = run_ocr_job(
    data_source="https://example.com/image.jpg",
    output_dataset="web-ocr",
    source_type="web"
)

# Multiple URLs from file (one per line)
job = run_ocr_job(
    data_source="urls.txt",
    output_dataset="web-ocr",
    source_type="web"
)
```

### 📁 Image Directories

```python
job = run_ocr_job(
    data_source="path/to/images/",
    output_dataset="dir-ocr",
    source_type="directory",
    gpus="v100:2"
)
```

### 🤗 HuggingFace Datasets

```python
job = run_ocr_job(
    data_source="davanstrien/ufo-ColPali",
    output_dataset="ufo-ocr",
    source_type="hf",
    gpus="a100:2"
)
```

## Advanced Usage

### Custom Jobs

Run any Python script on the cluster:

```python
from yale import run_job

script = """
import torch
from datasets import load_from_disk

# Load prepared dataset
dataset = load_from_disk("dataset")

print(f"Processing {len(dataset)} samples...")
# Your custom processing here
"""

job = run_job(
    script=script,
    data_source="path/to/data",
    source_type="auto",
    job_name="custom-job",
    gpus="v100:2",
    cpus_per_task=4,
    time_limit="02:00:00",
    memory="32G"
)
```

### Object-Oriented API

```python
from yale import YaleJobs

# Create SDK instance
yale = YaleJobs(config_path="config.yaml")
yale.connect()

# Submit job
job = yale.submit_job(
    script=my_script,
    data_source="path/to/data",
    job_name="my-job",
    gpus="v100:2"
)

# Monitor status
status = yale.get_job_status(job.job_id)
print(status)

# Close connection
yale.close()

# Or use context manager
with YaleJobs() as yale:
    yale.connect()
    job = yale.submit_job(...)
```

### Direct Data Source Usage

Work with data sources directly:

```python
from yale.data import PDFDataSource, IIIFDataSource

# Convert PDFs to images
pdf_ds = PDFDataSource("document.pdf")
images = pdf_ds.to_images(dpi=300)

# Create HuggingFace dataset
dataset = pdf_ds.to_dataset()
dataset.save_to_disk("pdf_dataset")

# IIIF manifest
iiif_ds = IIIFDataSource("https://example.com/manifest.json")
print(f"IIIF version: {iiif_ds.version}")
image_urls = iiif_ds.get_image_urls()
dataset = iiif_ds.to_dataset(max_size=2000)
```

## CLI Reference

### Job Commands

```bash
# Run custom script
yale jobs run script.py --data-source data/ --gpus v100:2

# Run OCR
yale jobs ocr <source> <output> [options]
  --source-type {auto,pdf,iiif,web,directory,hf}
  --model MODEL                    # Default: rednote-hilab/dots.ocr
  --batch-size N                   # Default: 16
  --max-samples N                  # Limit samples
  --gpus GPU_SPEC                  # Default: p100:2
  --partition PARTITION            # SLURM partition (default: gpu)
  --time HH:MM:SS                  # Time limit (default: 02:00:00)
  --env ENV_NAME                   # Conda environment (overrides config.yaml)
  --prompt-mode {ocr,layout-all,layout-only}  # DoTS.ocr mode (default: layout-all)
  --dataset-path PATH              # Use existing dataset on cluster (skips upload)
  --max-model-len N                # Maximum model context length (default: 32768)
  --max-tokens N                   # Maximum output tokens (default: 16384)
  --hpc-process                    # Process data on HPC (copy raw data first)
  --wait                           # Wait for completion

# Check status
yale jobs status <job-id>

# Cancel job
yale jobs cancel <job-id>

# Download results
yale jobs download --job-name NAME [--output-dir DIR] [--pattern PATTERN]

# View logs
yale jobs logs --job-name NAME
```

### GPU Options

Common GPU specifications:
- `p100:1` - Single P100 GPU
- `p100:2` - Two P100 GPUs
- `v100:1` - Single V100 GPU
- `v100:2` - Two V100 GPUs
- `a100:1` - Single A100 GPU

## How It Works

Yale Jobs handles the complete workflow:

1. **SSH Connection** - Connects to cluster with 2FA support
2. **Data Preparation** - Converts data from various sources to HuggingFace datasets
3. **Upload** - Transfers data to cluster via SFTP
4. **Job Creation** - Generates SLURM batch script
5. **Submission** - Submits job to SLURM queue
6. **Monitoring** - Tracks job status with `sacct`/`squeue`
7. **Results** - Downloads results when complete

## Examples

### Basic OCR with Different Prompt Modes

```bash
# Simple text extraction (ocr mode)
yale jobs ocr manuscript.pdf text-output \
    --source-type pdf \
    --prompt-mode ocr

# Full layout analysis with bounding boxes (layout-all - default)
# Note: layout-all uses a longer prompt, increase context if needed
yale jobs ocr documents/ layout-output \
    --source-type pdf \
    --prompt-mode layout-all \
    --gpus h200:1 \
    --partition gpu_h200 \
    --max-model-len 32768

# Layout structure only (no text content)
yale jobs ocr scans.pdf layout-only-output \
    --source-type pdf \
    --prompt-mode layout-only
```

**Note on Context Length:**
- **Simple OCR mode**: Default 32768 is usually enough
- **Layout-all mode**: May need 32768+ for complex/large images (default)
- **Error "decoder prompt too long"**: Increase `--max-model-len` (e.g., 49152 or 65536)
- DoTS.ocr supports up to ~128K tokens depending on available GPU memory

### Processing IIIF Manifest Lists

Process multiple IIIF manifests from a text file:

```bash
# Create a text file with manifest URLs (one per line)
cat > manifests.txt <<EOF
https://collections.library.yale.edu/manifests/11781249
https://collections.library.yale.edu/manifests/11781250
https://collections.library.yale.edu/manifests/11781251
EOF

# Default: Downloads images locally, then uploads dataset to cluster
yale jobs ocr manifests.txt output \
    --batch-size 16 \
    --gpus h200:1 \
    --partition gpu_h200

# With --hpc-process: Downloads images ON THE CLUSTER (recommended!)
yale jobs ocr manifests.txt output \
    --source-type iiif-list \
    --hpc-process \
    --batch-size 16 \
    --gpus h200:1 \
    --partition gpu_h200
```

**Without `--hpc-process` (default):**
1. Load IIIF manifests on your local machine
2. Download images from IIIF servers to your local machine
3. Convert to HuggingFace dataset locally
4. Upload entire dataset to cluster
5. Run OCR

**With `--hpc-process` (recommended for IIIF):**
1. Upload manifest list to cluster
2. Cluster loads IIIF manifests
3. **Cluster downloads images directly from IIIF servers** (faster, doesn't use your bandwidth!)
4. Cluster converts to HuggingFace dataset
5. Run OCR

💡 **Always use `--hpc-process` with IIIF manifests** - the cluster has better bandwidth to IIIF servers!

### HPC Data Processing

For large datasets or when bandwidth is limited, process data on the cluster instead of locally:

```bash
# Process PDF on HPC (copies raw PDF, processes there)
yale jobs ocr large-document.pdf output \
    --source-type pdf \
    --hpc-process \
    --gpus h200:1 \
    --partition gpu_h200

# Process directory of images on HPC
yale jobs ocr /local/images/ output \
    --source-type directory \
    --hpc-process
```

**How `--hpc-process` works:**
1. **Without flag** (default): Data is processed locally, converted to HuggingFace Dataset, then uploaded
2. **With flag**: Raw data is copied to cluster, preprocessing script runs there, then OCR runs

**When to use:**
- ✅ **IIIF manifests** (cluster has better bandwidth to IIIF servers!)
- ✅ Large PDFs or image directories (faster than local processing + upload)
- ✅ Slow local connection to cluster
- ✅ Want to leverage cluster's faster processing

**When NOT to use:**
- ❌ Small datasets (local processing is fine)
- ❌ HuggingFace datasets (already remote)

### Reusing Existing Datasets

Skip data upload when rerunning OCR on an existing dataset:

```bash
# First run - uploads data
yale jobs ocr manuscript.pdf first-output \
    --source-type pdf \
    --prompt-mode ocr \
    --job-name ocr-run-1

# Second run - reuse the uploaded dataset with different prompt
yale jobs ocr dummy.pdf second-output \
    --dataset-path /path/to/cluster/first-output_data \
    --prompt-mode layout-all \
    --job-name ocr-run-2
```

### More Examples

See the `examples/` directory for more:
- `simple_ocr.py` - Basic OCR usage
- `iiif_ocr.py` - IIIF manifest processing
- `custom_job.py` - Custom script execution
- `data_sources.py` - Working with different data sources

## Configuration

The `config.yaml` file supports:

```yaml
alias: cluster-hostname    # Required: SSH hostname
login: true                 # Optional: Login node
env: conda-env-name        # Optional: Conda environment
job_dir: path/to/jobs      # Optional: Job directory (supports wildcards)
result_dir: path/to/results # Optional: Results directory
2fa: true                  # Optional: Enable 2FA (default: true)
```

## Troubleshooting

### Connection Issues

If you have trouble connecting:
```bash
# Test SSH manually
ssh your-netid@yale-cluster

# Check config
cat config.yaml
```

### 2FA Issues

The system prompts for 2FA code after initial password. If this doesn't work, you may need to configure SSH keys.

### Job Not Starting

Check job status:
```bash
yale jobs status <job-id>
```

Check logs:
```bash
yale jobs logs --job-name <job-name>
```

## Comparison to HuggingFace Jobs

| Feature | HuggingFace Jobs | Yale Jobs |
|---------|------------------|-----------|
| Remote execution | ✅ HF infrastructure | ✅ Yale HPC |
| GPU support | ✅ | ✅ |
| Data sources | HF datasets | PDFs, IIIF, directories, web, HF |
| Authentication | HF token | SSH + 2FA |
| Job monitoring | ✅ | ✅ |
| Python SDK | ✅ | ✅ |
| CLI | ✅ | ✅ |

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please open an issue or PR.

## Support

For issues or questions:
- Open a GitHub issue
- Contact: [your-email]
