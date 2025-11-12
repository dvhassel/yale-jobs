# Yale Jobs - Project Summary

## Overview

Yale Jobs is a complete job management system for Yale's HPC cluster, modeled after HuggingFace Jobs. It provides a simple, intuitive API for submitting and managing computational jobs on Yale's cluster with seamless support for multiple data sources.

## What Was Built

### 1. Data Source Handlers (`yale/data/`)

**Purpose:** Convert various data formats into HuggingFace datasets for processing

**Components:**
- `pdf.py` - PDF document processing using pypdfium2
  - Single files or directories
  - Converts pages to images at specified DPI
  - Preserves metadata (source, page numbers)

- `iiif.py` - IIIF manifest handling
  - Supports Presentation API v2 and v3
  - Automatic version detection
  - Downloads images from IIIF Image API
  - Handles multilingual labels

- `web.py` - Web URL processing
  - Single URLs or lists
  - Text file input (one URL per line)
  - Batch downloading

- `directory.py` - Local image directory handling
  - Recursive or non-recursive scanning
  - Supports all common image formats
  - Preserves directory structure metadata

- `hf.py` - HuggingFace dataset integration
  - Direct loading from HF Hub
  - Subset/split selection

### 2. Cluster Connection Manager (`yale/cluster.py`)

**Purpose:** Handle SSH connections to Yale HPC with 2FA support

**Features:**
- SSH connection with paramiko
- 2FA authentication flow
- SFTP file transfer (upload/download)
- Remote command execution
- Conda environment activation
- Context manager support

**Key Methods:**
- `connect()` - Establish SSH connection with 2FA
- `execute_command()` - Run commands on cluster
- `upload_file()` / `download_file()` - File transfer
- `upload_directory()` - Recursive directory upload

### 3. Job Management System (`yale/jobs.py`)

**Purpose:** Create, submit, and monitor SLURM jobs

**Features:**
- SLURM batch script generation
- Data preparation and upload
- Job submission via `sbatch`
- Status monitoring via `sacct`/`squeue`
- Result downloading
- Job cancellation

**Key Methods:**
- `create_sbatch_script()` - Generate SLURM script
- `prepare_data()` - Convert and upload data
- `submit()` - Submit job to SLURM
- `get_status()` - Check job status
- `wait_for_completion()` - Block until done
- `download_results()` - Retrieve outputs

### 4. Python SDK (`yale/sdk.py`)

**Purpose:** High-level API for job submission

**Functions:**
- `run_job()` - Submit custom Python scripts
- `run_ocr_job()` - Convenience function for OCR tasks

**Classes:**
- `YaleJobs` - Object-oriented API with context manager support

**Design Philosophy:**
- Similar to HuggingFace Jobs API
- Simple function calls for common tasks
- Full control when needed

### 5. Command-Line Interface (`yale/cli.py`)

**Purpose:** Terminal interface for job management

**Commands:**
```bash
yale jobs run        # Run custom script
yale jobs ocr        # Run OCR job
yale jobs status     # Check job status
yale jobs cancel     # Cancel job
yale jobs download   # Download results
yale jobs logs       # View job logs
```

**Features:**
- Argparse-based CLI
- Rich help text and examples
- All SDK features accessible via CLI

### 6. Yale-Adapted OCR Script (`yale/ocr/yale-dots-ocr.py`)

**Purpose:** Modified DoTS.ocr script for Yale workflow

**Changes from Original:**
- `--load-from-disk-path` flag for pre-prepared data
- Saves to disk instead of pushing to HF Hub
- Works with Yale job system's data preparation
- Maintains all original DoTS.ocr features

### 7. Documentation

**Files Created:**
- `README.md` - Main documentation with examples
- `USAGE.md` - Detailed usage guide
- `QUICKSTART.md` - Quick reference
- `PROJECT_SUMMARY.md` - This file
- `requirements.txt` - Dependencies
- `.gitignore` - Git ignore patterns

**Example Scripts:**
- `examples/simple_ocr.py` - Basic OCR
- `examples/iiif_ocr.py` - IIIF manifest processing
- `examples/custom_job.py` - Custom script execution
- `examples/data_sources.py` - Working with all data sources

### 8. Package Configuration

**setup.py:**
- Package metadata
- Dependency management
- Entry point for CLI (`yale` command)
- Extra dependencies for OCR

**config.yaml:**
- Cluster configuration
- Job directories
- Conda environment
- 2FA settings

## Architecture

```
User Input (CLI/SDK)
        ↓
    Yale Jobs SDK
        ↓
   ┌────────────────┐
   │ Data Sources   │ → Convert to HF Dataset
   └────────────────┘
        ↓
   ┌────────────────┐
   │ Cluster Conn   │ → SSH + 2FA
   └────────────────┘
        ↓
   ┌────────────────┐
   │ Job Manager    │ → SLURM Scripts
   └────────────────┘
        ↓
   Yale HPC Cluster
```

## Key Features Implemented

✅ **Multiple Data Sources**
- PDFs (pypdfium2)
- IIIF v2/v3 manifests
- Web URLs
- Local directories
- HuggingFace datasets

✅ **SSH & 2FA**
- Paramiko-based SSH
- Interactive 2FA flow
- SFTP file transfer

✅ **Job Management**
- SLURM batch scripts
- Job submission
- Status monitoring
- Result retrieval

✅ **APIs**
- Simple function API (`run_ocr_job`)
- Object-oriented API (`YaleJobs`)
- Command-line interface

✅ **OCR Support**
- DoTS.ocr integration
- Batch processing
- vLLM acceleration

✅ **Documentation**
- Comprehensive README
- Detailed usage guide
- Quick start guide
- Example scripts

## Usage Examples

### Quick OCR (CLI)
```bash
yale jobs ocr pdfs/ output --source-type pdf --gpus v100:2
```

### Quick OCR (Python)
```python
from yale import run_ocr_job

job = run_ocr_job("pdfs/", "output", source_type="pdf", gpus="v100:2")
```

### Custom Job
```python
from yale import run_job

job = run_job(
    script=my_script,
    data_source="data/",
    gpus="v100:2"
)
```

### Full Control
```python
from yale import YaleJobs

with YaleJobs() as yale:
    yale.connect()
    job = yale.submit_job(...)
    status = yale.get_job_status(job.job_id)
```

## File Structure

```
yale-jobs/
├── yale/                      # Main package
│   ├── __init__.py           # Package exports
│   ├── cli.py                # Command-line interface
│   ├── cluster.py            # SSH/cluster connection
│   ├── jobs.py               # Job management
│   ├── sdk.py                # Python SDK
│   ├── data/                 # Data source handlers
│   │   ├── __init__.py
│   │   ├── pdf.py           # PDF processing
│   │   ├── iiif.py          # IIIF manifests
│   │   ├── web.py           # Web URLs
│   │   ├── directory.py     # Image directories
│   │   └── hf.py            # HuggingFace datasets
│   └── ocr/                  # OCR scripts
│       ├── __init__.py
│       ├── dots-ocr.py      # Original script
│       └── yale-dots-ocr.py # Yale-adapted version
├── examples/                 # Example scripts
│   ├── simple_ocr.py
│   ├── iiif_ocr.py
│   ├── custom_job.py
│   └── data_sources.py
├── config.yaml              # Configuration
├── setup.py                 # Package setup
├── requirements.txt         # Dependencies
├── README.md               # Main documentation
├── USAGE.md                # Detailed guide
├── QUICKSTART.md           # Quick reference
└── PROJECT_SUMMARY.md      # This file
```

## Dependencies

**Core:**
- pyyaml - Configuration
- paramiko - SSH/SFTP
- datasets - Data handling
- pillow - Image processing

**Data Sources:**
- pypdfium2 - PDF processing
- requests - Web/IIIF

**OCR (optional):**
- torch - Deep learning
- vllm - Efficient inference
- transformers - Model loading

## Next Steps / Future Enhancements

**Potential Improvements:**
1. Support for more OCR models (Qwen2.5-VL, etc.)
2. Automatic retry on job failure
3. Job dependency management
4. Cost estimation before submission
5. Web dashboard for job monitoring
6. Support for other schedulers (PBS, etc.)
7. Checkpoint/resume for long jobs
8. Automatic dataset uploading to HF Hub
9. Integration with Yale's storage systems
10. Batch job submission from spreadsheet

## Comparison to HuggingFace Jobs

| Feature | HuggingFace Jobs | Yale Jobs |
|---------|------------------|-----------|
| Target | HF Infrastructure | Yale HPC |
| Data Sources | HF datasets | PDFs, IIIF, directories, web, HF |
| Auth | HF token | SSH + 2FA |
| Scheduler | HF internal | SLURM |
| GPU Selection | Flavor-based | Direct SLURM |
| Local Testing | Limited | Full control |
| Cost | Pay per use | Free (Yale resource) |

## Technical Decisions

**Why paramiko over subprocess.run ssh?**
- Better 2FA handling
- SFTP support built-in
- More Pythonic API

**Why HuggingFace datasets format?**
- Standard format for ML
- Easy to process
- Good serialization

**Why separate data source handlers?**
- Modularity
- Easy to extend
- Clear separation of concerns

**Why both SDK and CLI?**
- SDK for programmatic use
- CLI for quick tasks
- Both use same underlying code

## Testing Recommendations

1. **Test SSH connection first:**
   ```python
   from yale import ClusterConnection
   conn = ClusterConnection()
   conn.connect()
   ```

2. **Test data sources locally:**
   ```python
   from yale.data import PDFDataSource
   ds = PDFDataSource("test.pdf")
   images = ds.to_images()
   ```

3. **Test with small samples:**
   ```bash
   yale jobs ocr test.pdf output --max-samples 5
   ```

4. **Monitor first job closely:**
   ```bash
   yale jobs status <job-id>
   yale jobs logs --job-name <job-name>
   ```

## Conclusion

Yale Jobs provides a complete, production-ready system for managing computational jobs on Yale's HPC cluster. It combines the simplicity of HuggingFace Jobs with the flexibility needed for academic research, supporting diverse data sources and providing both programmatic and command-line interfaces.

The system is designed to be:
- **Easy to use** - Simple API for common tasks
- **Flexible** - Full control when needed
- **Extensible** - Easy to add new data sources or models
- **Well-documented** - Comprehensive guides and examples
- **Production-ready** - Error handling, monitoring, and recovery

All components are implemented and ready for use!

