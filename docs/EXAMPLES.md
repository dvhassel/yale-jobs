# Yale Jobs - Usage Examples

## 🎯 Quick Start

### Basic OCR on a PDF

```bash
yale jobs ocr document.pdf output \
    --source-type pdf \
    --gpus h200:1 \
    --partition gpu_h200
```

## 🗂️ IIIF Manifest Lists

### Process Multiple Manifests from a Text File

```bash
# Your manifests.txt file:
# https://collections.library.yale.edu/manifests/11781249
# https://collections.library.yale.edu/manifests/11781250
# https://collections.library.yale.edu/manifests/11781251

# RECOMMENDED: Download images on the cluster (faster!)
yale jobs ocr manifests.txt yale-focus-scans \
    --hpc-process \
    --batch-size 16 \
    --gpus h200:1 \
    --partition gpu_h200 \
    --prompt-mode layout-all
```

**What happens with `--hpc-process`:**
1. System auto-detects `.txt` file with IIIF URLs
2. Uploads manifest list to cluster
3. **Cluster loads manifests and downloads images** (uses cluster's faster network!)
4. Cluster combines into single dataset
5. Runs OCR on all images

**Without `--hpc-process`:**
- Downloads images to YOUR local machine first
- Then uploads dataset to cluster
- Slower and uses your bandwidth

### Manual Source Type

```bash
# Explicitly specify iiif-list if auto-detection doesn't work
yale jobs ocr manifests.txt output \
    --source-type iiif-list \
    --batch-size 16
```

## 🖥️ HPC Data Processing

### When to Use `--hpc-process`

**Use it for:**
- **IIIF manifests** (cluster downloads images directly - much faster!)
- Large PDFs (100+ pages)
- Directories with thousands of images
- Slow local internet connection
- Want to leverage cluster's faster CPU/storage

**Don't use it for:**
- Small datasets (< 100 images)
- HuggingFace datasets (already remote)

### Process Large PDF on HPC

```bash
# Without --hpc-process (default):
# 1. Local: Convert PDF → images → dataset
# 2. Upload entire dataset to cluster
# 3. Cluster: Run OCR

# With --hpc-process:
# 1. Upload raw PDF to cluster
# 2. Cluster: Convert PDF → images → dataset
# 3. Cluster: Run OCR

yale jobs ocr large-manuscript.pdf output \
    --source-type pdf \
    --hpc-process \
    --gpus h200:1 \
    --partition gpu_h200 \
    --time 04:00:00
```

### Process Image Directory on HPC

```bash
# Process 10,000 images - let the cluster do the work
yale jobs ocr /local/scans/ output \
    --source-type directory \
    --hpc-process \
    --batch-size 32 \
    --gpus h200:2 \
    --partition gpu_h200
```

## 🔄 Combining Features

### IIIF Manifests + Custom Settings (Cluster Processing)

```bash
# Process IIIF manifests on cluster for best performance
yale jobs ocr manifests.txt yale-collection \
    --source-type iiif-list \
    --hpc-process \
    --batch-size 32 \
    --gpus h200:2 \
    --partition gpu_h200 \
    --time 03:00:00 \
    --env vllm \
    --prompt-mode layout-all \
    --max-model-len 49152
```

### Reusing Dataset with Different Prompt Mode

```bash
# First run with layout-all
yale jobs ocr document.pdf output-layout \
    --prompt-mode layout-all

# Get the dataset path from the job output
# (e.g., /home/user/project/shared/test/yale-ocr_data)

# Rerun with simple OCR mode (no data upload)
yale jobs ocr document.pdf output-text \
    --prompt-mode ocr \
    --dataset-path /home/user/project/shared/test/yale-ocr_data
```

### HPC Processing + Large Context

```bash
# Large, complex documents - process on HPC with large context
yale jobs ocr large-technical-manual.pdf output \
    --source-type pdf \
    --hpc-process \
    --prompt-mode layout-all \
    --max-model-len 65536 \
    --max-tokens 32768 \
    --batch-size 4 \
    --gpus h200:1 \
    --partition gpu_h200 \
    --time 08:00:00
```

## 📊 Real-World Workflows

### Batch Processing Multiple Collections

```bash
# Create manifest lists for each collection
ls collections/*.txt

# collections/focus-scans.txt
# collections/manuscripts.txt
# collections/rare-books.txt

# Process each collection
for list in collections/*.txt; do
    name=$(basename "$list" .txt)
    yale jobs ocr "$list" "yale-$name" \
        --source-type iiif-list \
        --batch-size 32 \
        --gpus h200:2 \
        --partition gpu_h200
done
```

### Mixed Data Sources

```bash
# Process PDFs on HPC
yale jobs ocr /local/pdfs/ output-pdfs \
    --source-type pdf \
    --hpc-process

# Process IIIF manifests (always remote)
yale jobs ocr manifests.txt output-iiif \
    --source-type iiif-list

# Check status of both
yale jobs status <pdf-job-id>
yale jobs status <iiif-job-id>
```

## 🔧 Advanced Configuration

### Custom Environment and Resources

```bash
yale jobs ocr data/ output \
    --env my-custom-env \
    --gpus a100:4 \
    --partition gpu_a100 \
    --time 12:00:00 \
    --batch-size 64 \
    --max-model-len 131072
```

### Testing with Limited Samples

```bash
# Test with first 10 samples
yale jobs ocr manifests.txt test-output \
    --max-samples 10 \
    --batch-size 2 \
    --time 00:30:00
```

## 💡 Tips and Best Practices

1. **Start Small**: Test with `--max-samples 10` first
2. **Monitor Context Length**: Watch for "decoder prompt too long" errors
3. **Batch Size**: Increase for A100/H200 GPUs (32-64), decrease for smaller GPUs (4-16)
4. **Time Limits**: Add extra time for preprocessing when using `--hpc-process`
5. **Manifest Lists**: One URL per line, empty lines and `#` comments are ignored
6. **Reuse Datasets**: Save time by using `--dataset-path` for different prompt modes
7. **Dependencies**: `uv` handles all Python dependencies automatically - no manual setup!
8. **Resolution**: Lower `--max-resolution` for faster downloads and less memory usage

## 🚨 Common Patterns

### "I have a bunch of IIIF URLs"

```bash
# Put them in a .txt file
echo "https://collections.library.yale.edu/manifests/11781249" > my-manifests.txt
echo "https://collections.library.yale.edu/manifests/11781250" >> my-manifests.txt

# Use --hpc-process so cluster downloads the images (much faster!)
yale jobs ocr my-manifests.txt output --hpc-process
```

### "My PDFs are huge and local processing is slow"

```bash
yale jobs ocr big-pdf.pdf output --hpc-process
```

### "I want both text and layout analysis"

```bash
# Run twice with same dataset
yale jobs ocr doc.pdf output-text --prompt-mode ocr
yale jobs ocr doc.pdf output-layout --prompt-mode layout-all \
    --dataset-path /path/from/first/run
```

### "Context length error!"

```bash
# Increase context window
yale jobs ocr doc.pdf output \
    --max-model-len 65536 \
    --max-tokens 32768
```

