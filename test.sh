yale jobs ocr manifests.txt kissinger \
    --batch-size 16 \
    --gpus h200:1 \
    --time 05:00:00 \
    --partition gpu_h200 \
    --prompt-mode layout-all \
    --hpc-process \
    --max-model-len 32768 \
    --max-resolution 1500 \
    --job-name kissinger
