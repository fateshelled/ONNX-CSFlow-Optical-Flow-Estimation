if [ -z "$1" ]; then
    echo "Usage: $0 <model_path> <save_model_path> <workspace>"
    echo "WORKSPACE : GPU memory workspace. ex. 3221225472 (3GB), $((1024*1024*1024)) (1GB)"
    exit 1
fi

MODEL_PATH=$1
SAVE_MODEL_PATH=$2
TRT_WORKSPACE=$3

echo "Model Path: ${MODEL_PATH}"
echo "Save Model Path: ${SAVE_MODEL_PATH}"
echo "Workspace size: ${TRT_WORKSPACE}"
echo ""

trtexec \
    --onnx=${MODEL_PATH} \
    --saveEngine=${SAVE_MODEL_PATH} \
    --fp16 \
    --verbose \
    --workspace=${TRT_WORKSPACE}
