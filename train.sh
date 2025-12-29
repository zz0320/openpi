#!/bin/bash

# =================================================================================
# Main parameters for training
# =================================================================================
# The ID of the data repository
REPO_ID="merge_labor01_0808"
# The name of the training configuration
CONFIG_NAME="pi05_qingloong_low_mem_finetune"
# The name for this experiment run
EXP_NAME="test0912"
# Memory fraction for XLA
XLA_MEM_FRACTION="0.95"
# =================================================================================

echo "Running with the following configuration:"
echo "REPO_ID: $REPO_ID"
echo "CONFIG_NAME: $CONFIG_NAME"
echo "EXP_NAME: $EXP_NAME"
echo "XLA_MEM_FRACTION: $XLA_MEM_FRACTION"
echo "================================================================================="

echo "Step 1: Computing normalization stats..."
uv run scripts/compute_norm_stats.py --config-name "$CONFIG_NAME"

ASSET_PATH="/workspace/code/openpi/assets/$CONFIG_NAME"

echo "Step 2: Updating normalization stats for qingloong..."
rm -f "$ASSET_PATH/qingloong/norm_stats.json"
cp "$ASSET_PATH/$REPO_ID/norm_stats.json" "$ASSET_PATH/qingloong/norm_stats.json"

echo "Step 3: Starting training..."
XLA_PYTHON_CLIENT_MEM_FRACTION="$XLA_MEM_FRACTION" uv run /workspace/code/openpi/scripts/train.py "$CONFIG_NAME" --exp-name="$EXP_NAME" --overwrite

echo "Training script finished."