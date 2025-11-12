# Yale Jobs Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                           │
├─────────────────────────────────┬───────────────────────────────┤
│         CLI (cli.py)            │      Python SDK (sdk.py)      │
│  $ yale jobs ocr ...            │  from yale import run_ocr_job │
└─────────────────────────────────┴───────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      CORE COMPONENTS                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌───────────────────┐  ┌───────────────────┐  ┌──────────────┐│
│  │ Data Sources      │  │ Job Manager       │  │ Cluster Conn ││
│  │ (data/)           │  │ (jobs.py)         │  │ (cluster.py) ││
│  ├───────────────────┤  ├───────────────────┤  ├──────────────┤│
│  │ • PDFDataSource   │  │ • SLURM scripts   │  │ • SSH client ││
│  │ • IIIFDataSource  │  │ • Job submission  │  │ • 2FA flow   ││
│  │ • WebDataSource   │  │ • Status monitor  │  │ • SFTP xfer  ││
│  │ • DirectoryDS     │  │ • Result download │  │ • Commands   ││
│  │ • HFDataSource    │  └───────────────────┘  └──────────────┘│
│  └───────────────────┘                                           │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    YALE HPC CLUSTER                              │
├─────────────────────────────────────────────────────────────────┤
│  • SLURM Scheduler                                               │
│  • GPU Nodes (P100, V100, A100)                                 │
│  • Shared Storage                                                │
│  • Conda Environments                                            │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. OCR Job Submission

```
User Input
   ↓
yale jobs ocr pdfs/ output --source-type pdf
   ↓
CLI Parser (cli.py)
   ↓
run_ocr_job() (sdk.py)
   ↓
┌─────────────────────────────────────────┐
│ 1. Data Preparation                     │
│    PDFDataSource("pdfs/")               │
│    → to_dataset()                       │
│    → save_to_disk()                     │
└─────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────┐
│ 2. Connection                           │
│    ClusterConnection()                  │
│    → connect() [SSH + 2FA]              │
└─────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────┐
│ 3. Upload                               │
│    connection.upload_directory()        │
│    → Transfer dataset to cluster        │
└─────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────┐
│ 4. Job Creation                         │
│    YaleJob.create_sbatch_script()       │
│    → Generate SLURM script              │
└─────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────┐
│ 5. Submission                           │
│    job.submit()                         │
│    → sbatch script.sh                   │
└─────────────────────────────────────────┘
   ↓
SLURM Queue → GPU Node → Processing
   ↓
Results in cluster storage
   ↓
┌─────────────────────────────────────────┐
│ 6. Download (optional)                  │
│    job.download_results()               │
│    → SFTP transfer                      │
└─────────────────────────────────────────┘
   ↓
Local results/
```

### 2. Data Source Processing

```
Input Source
   ↓
┌──────────────────────────────────────────────────────┐
│                  Data Source Handler                  │
├──────────────────────────────────────────────────────┤
│                                                       │
│  PDF: document.pdf                                    │
│   → PDFDataSource                                     │
│   → PdfDocument.render()                              │
│   → PIL Images                                        │
│                                                       │
│  IIIF: manifest.json                                  │
│   → IIIFDataSource                                    │
│   → Parse v2/v3                                       │
│   → Extract URLs                                      │
│   → Download images                                   │
│                                                       │
│  Web: urls.txt                                        │
│   → WebDataSource                                     │
│   → requests.get()                                    │
│   → PIL Images                                        │
│                                                       │
│  Directory: images/                                   │
│   → DirectoryDataSource                               │
│   → Glob for image files                              │
│   → Load with PIL                                     │
│                                                       │
│  HuggingFace: dataset-name                            │
│   → HFDataSource                                      │
│   → load_dataset()                                    │
│                                                       │
└──────────────────────────────────────────────────────┘
   ↓
HuggingFace Dataset
   {
     "image": [PIL.Image, ...],
     "metadata": {...}
   }
   ↓
Save to disk / Upload to cluster
```

## Component Details

### Data Sources Module (`yale/data/`)

**Purpose:** Convert various data formats to HuggingFace datasets

**Flow:**
```python
DataSource(input)
   .to_images()    # → List[dict] with PIL images
   .to_dataset()   # → HuggingFace Dataset
```

**Implementations:**
- `PDFDataSource` - pypdfium2 rendering
- `IIIFDataSource` - IIIF API client
- `WebDataSource` - HTTP downloader
- `DirectoryDataSource` - File system scanner
- `HFDataSource` - HF Hub loader

### Cluster Connection (`yale/cluster.py`)

**Purpose:** SSH/SFTP connection manager

**Components:**
```python
ClusterConnection(config_path)
   .connect(username, password)
      → SSH handshake
      → 2FA prompt & verification
      → SFTP initialization
   
   .execute_command(cmd)
      → Channel.exec_command()
      → Return stdout/stderr/exit_code
   
   .upload_file(local, remote)
      → SFTP.put()
   
   .download_file(remote, local)
      → SFTP.get()
```

### Job Manager (`yale/jobs.py`)

**Purpose:** SLURM job lifecycle management

**Workflow:**
```python
YaleJob(connection, job_name)
   .prepare_data(source)
      → Convert to dataset
      → Upload to cluster
   
   .create_sbatch_script(script)
      → Generate SLURM headers
      → Add conda activation
      → Add script content
   
   .submit(script_path)
      → sbatch command
      → Extract job ID
   
   .get_status()
      → sacct/squeue
      → Parse state
   
   .wait_for_completion()
      → Poll until done
   
   .download_results()
      → SFTP transfer
```

### Python SDK (`yale/sdk.py`)

**Purpose:** High-level API

**Functions:**
```python
run_job(script, data_source, ...)
   → Full job submission flow
   → Returns YaleJob instance

run_ocr_job(data_source, output, ...)
   → OCR-specific wrapper
   → Pre-built OCR script
   → Batch processing config

YaleJobs class
   → Object-oriented interface
   → Context manager support
   → Multiple job management
```

### CLI (`yale/cli.py`)

**Purpose:** Command-line interface

**Commands:**
```
yale jobs run <script>
   → Custom script execution

yale jobs ocr <source> <output>
   → OCR job submission

yale jobs status <job-id>
   → Job status check

yale jobs cancel <job-id>
   → Job cancellation

yale jobs download --job-name <name>
   → Result download

yale jobs logs --job-name <name>
   → Log viewing
```

## Security Considerations

### SSH Authentication
```
1. Username/password prompt
2. SSH key exchange
3. 2FA code prompt
4. Session establishment
5. Keep-alive heartbeat
```

### Data Transfer
```
1. SFTP over SSH
2. Encrypted transfer
3. File permissions preserved
4. Checksum verification (implicit)
```

### Configuration
```yaml
# config.yaml should be in .gitignore
# Contains cluster hostname
# May contain sensitive paths
```

## Error Handling

### Connection Errors
```python
try:
    connection.connect()
except paramiko.AuthenticationException:
    → Prompt for credentials again
except socket.timeout:
    → Retry with backoff
except Exception as e:
    → Log and raise
```

### Job Errors
```python
job.submit(script_path)
   → Check exit_code
   → Parse stderr for errors
   → Raise RuntimeError if failed

job.get_status()
   → Handle UNKNOWN state
   → Check both sacct and squeue
   → Fallback to log files
```

### Data Transfer Errors
```python
connection.upload_file(local, remote)
   → Check file exists locally
   → Ensure remote dir exists
   → Verify upload completion
   → Retry on network error
```

## Performance Considerations

### Batch Processing
- Use vLLM for GPU efficiency
- Adjust batch size for GPU memory
- Monitor GPU utilization

### Data Transfer
- Compress before upload if possible
- Use parallel transfers for multiple files
- Consider cluster-local data sources

### Job Scheduling
- Choose appropriate partition
- Request accurate time limits
- Use job arrays for multiple tasks

## Extension Points

### Adding New Data Sources
```python
# yale/data/new_source.py
class NewDataSource:
    def __init__(self, source):
        self.source = source
    
    def to_images(self):
        # Convert to PIL images
        pass
    
    def to_dataset(self):
        # Return HF Dataset
        pass
```

### Adding New OCR Models
```python
# yale/ocr/new-model.py
# Adapt script similar to yale-dots-ocr.py
# Update CLI to support new model
# Add model-specific parameters
```

### Custom Job Types
```python
# Use run_job() with custom script
# Or extend YaleJob class
# Or create new convenience function
```

## Deployment

### Installation
```bash
git clone repo
cd yale-jobs
pip install -e .
```

### Configuration
```bash
cp config.yaml.example config.yaml
# Edit with cluster details
```

### Testing
```bash
# Test connection
python -c "from yale import ClusterConnection; c = ClusterConnection(); c.connect()"

# Test data source
python -c "from yale.data import PDFDataSource; ds = PDFDataSource('test.pdf'); print(len(ds.to_images()))"

# Test small job
yale jobs ocr test.pdf output --max-samples 1
```

## Monitoring

### Job Status
```bash
# Real-time
watch -n 10 "yale jobs status <job-id>"

# Logs
tail -f job.out  # On cluster
```

### Resource Usage
```bash
# On cluster
squeue -u $USER
sacct -j <job-id> --format=JobID,Elapsed,State,MaxRSS,AllocCPUS
```

## Best Practices

1. **Test locally first** - Verify data sources and scripts
2. **Start small** - Use --max-samples for testing
3. **Monitor closely** - Check first few jobs carefully
4. **Use version control** - Track scripts and configs
5. **Document parameters** - Note what worked
6. **Clean up** - Remove old job files and datasets
7. **Batch similar jobs** - More efficient than many small jobs
8. **Choose right GPU** - Match GPU to task requirements
9. **Estimate time** - Add buffer to time limits
10. **Check quotas** - Monitor storage and compute usage

---

This architecture enables flexible, efficient job submission to Yale's HPC cluster with a simple, intuitive API similar to HuggingFace Jobs while supporting diverse data sources and providing full control over job execution.

