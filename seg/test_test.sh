# Purpose: get test predictions for public leaderboard
TEST_ROOT=$1
CONFIG_FILE="${TEST_ROOT}/*${TEST_ROOT: -1}.py" # or .json for old configs
CHECKPOINT_FILE="${TEST_ROOT}/best_mIoU_iter_*.pth"
SAVE_PATH="${TEST_ROOT}/labelTrainIds"

# Debug: Print the paths to check if they're correct
echo "Config file: ${CONFIG_FILE}"
echo "Checkpoint file: ${CHECKPOINT_FILE}"

python -m tools.test \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    --test-set \
    --format-only \
    --eval-option imgfile_prefix=${SAVE_PATH} to_label_id=False