#!/bin/bash

###############################################################################
# Parallel Video Inference Script
#
# This script runs parallel video inference using DeepLabCut across multiple
# GPUs with CPU affinity control.
#
# Usage examples:
#   # Use default settings (25 workers, GPUs 4,5,6, 10 videos per batch)
#   ./run_parallel_inference.sh
#
#   # Customize settings
#   ./run_parallel_inference.sh --num-workers 20 --gpus 4,5 --videos-per-batch 5
#
#   # Use specific GPUs and different batch size
#   ./run_parallel_inference.sh --gpus 0,1,2,3 --batch-size 128
###############################################################################

# Default configuration (edit these or override with command-line args)
VIDEO_DIR="/home/hwjwei/scratch/hwjwei/pia02/DLC_MODELS/general/interosseous_pn24-x-2025-10-24/videos"
CONFIG_PATH="/home/hwjwei/scratch/hwjwei/pia02/DLC_MODELS/general/interosseous_pn24-x-2025-10-24/config.yaml"
NUM_WORKERS=4
GPUS="4,5"
CORES_PER_WORKER=3
VIDEOS_PER_BATCH=10
BATCH_SIZE=64
ITERATION_NUM=0
CREATE_VIDEO=""  # Empty means no flag (disabled), set to "--create-video" to enable
LOG_DIR="./logs"

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_SCRIPT="${SCRIPT_DIR}/parallel_video_inference.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Print header
echo "=============================================================================="
echo "  Parallel Video Inference for DeepLabCut"
echo "=============================================================================="
echo ""

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    print_error "Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Parse command line arguments (override defaults)
while [[ $# -gt 0 ]]; do
    case $1 in
        --video-dir)
            VIDEO_DIR="$2"
            shift 2
            ;;
        --config-path)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --cores-per-worker)
            CORES_PER_WORKER="$2"
            shift 2
            ;;
        --videos-per-batch)
            VIDEOS_PER_BATCH="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --iteration-num)
            ITERATION_NUM="$2"
            shift 2
            ;;
        --create-video)
            CREATE_VIDEO="--create-video"
            shift
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --video-dir PATH          Video directory (default: $VIDEO_DIR)"
            echo "  --config-path PATH        DLC config.yaml path (default: $CONFIG_PATH)"
            echo "  --num-workers N           Number of workers (default: $NUM_WORKERS)"
            echo "  --gpus GPUS              Comma-separated GPU IDs (default: $GPUS)"
            echo "  --cores-per-worker N      CPU cores per worker (default: $CORES_PER_WORKER)"
            echo "  --videos-per-batch N      Videos per batch (default: $VIDEOS_PER_BATCH)"
            echo "  --batch-size N            Inference batch size (default: $BATCH_SIZE)"
            echo "  --iteration-num N         DLC iteration (default: $ITERATION_NUM)"
            echo "  --create-video            Create labeled videos (disabled by default)"
            echo "  --log-dir PATH            Log directory (default: $LOG_DIR)"
            echo "  -h, --help                Show this help message"
            echo ""
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate configuration
print_info "Validating configuration..."

if [ ! -d "$VIDEO_DIR" ]; then
    print_error "Video directory not found: $VIDEO_DIR"
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    print_error "Config file not found: $CONFIG_PATH"
    exit 1
fi

# Count videos
VIDEO_COUNT=$(find "$VIDEO_DIR" -maxdepth 1 -name "*.mp4" | wc -l)
if [ "$VIDEO_COUNT" -eq 0 ]; then
    print_error "No .mp4 videos found in $VIDEO_DIR"
    exit 1
fi

print_success "Found $VIDEO_COUNT videos to process"

# Display configuration
echo ""
print_info "Configuration:"
echo "  Video directory:     $VIDEO_DIR"
echo "  Config path:         $CONFIG_PATH"
echo "  Number of workers:   $NUM_WORKERS"
echo "  GPUs:                $GPUS"
echo "  Cores per worker:    $CORES_PER_WORKER"
echo "  Videos per batch:    $VIDEOS_PER_BATCH"
echo "  Inference batch size: $BATCH_SIZE"
echo "  DLC iteration:       $ITERATION_NUM"
echo "  Create videos:       $([ -z "$CREATE_VIDEO" ] && echo "No" || echo "Yes")"
echo "  Log directory:       $LOG_DIR"
echo ""

# Calculate resource usage
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}
TOTAL_CORES=$((NUM_WORKERS * CORES_PER_WORKER))
WORKERS_PER_GPU=$((NUM_WORKERS / NUM_GPUS))

print_info "Resource allocation:"
echo "  Total GPUs:          $NUM_GPUS"
echo "  Workers per GPU:     ~$WORKERS_PER_GPU"
echo "  Total CPU cores:     $TOTAL_CORES"
echo ""

# Confirm before starting
read -p "Start processing? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_warning "Cancelled by user"
    exit 0
fi

# Create log directory
mkdir -p "$LOG_DIR"

# Build command
CMD="python3 $PYTHON_SCRIPT \
    --video-dir \"$VIDEO_DIR\" \
    --config-path \"$CONFIG_PATH\" \
    --num-workers $NUM_WORKERS \
    --gpus $GPUS \
    --cores-per-worker $CORES_PER_WORKER \
    --videos-per-batch $VIDEOS_PER_BATCH \
    --batch-size $BATCH_SIZE \
    --iteration-num $ITERATION_NUM \
    --log-dir \"$LOG_DIR\" \
    $CREATE_VIDEO"

# Show command
print_info "Running command:"
echo "$CMD"
echo ""
echo "=============================================================================="
echo ""

# Run the Python script
START_TIME=$(date +%s)

eval $CMD
EXIT_CODE=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

echo ""
echo "=============================================================================="

if [ $EXIT_CODE -eq 0 ]; then
    print_success "Processing completed successfully!"
    echo "  Total time: ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
    echo "  Logs saved to: $LOG_DIR"
else
    print_error "Processing failed with exit code $EXIT_CODE"
    echo "  Check logs in: $LOG_DIR"
fi

echo "=============================================================================="

exit $EXIT_CODE
