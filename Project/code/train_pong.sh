#!/bin/zsh
# Train Pong DQN variants - each in its own Terminal window
# Usage: ./train_pong.sh [OPTIONS] [variants...]
# Examples:
#   ./train_pong.sh                              # Train all 5 variants with defaults
#   ./train_pong.sh dqn ddqn                     # Train only DQN and Double DQN
#   ./train_pong.sh --episodes 2000 ddqn_duel    # Train with custom episodes
#   ./train_pong.sh --lr 1e-4 --epsilon-decay 0.999  # Custom hyperparams for all

# Default hyperparameters (tuned for 18-action space)
EPISODES=1000
BATCH_SIZE=64
LR="1e-4"
BUFFER_SIZE=100000
EPSILON_START=1.0
EPSILON_END=0.01
EPSILON_DECAY=0.999
GAMMA=0.99
TARGET_UPDATE=1
UPDATE_EVERY=4

CODE_DIR="/Users/bittu/Desktop/GitHub/nn_class/Project/code"
CONDA_ENV="rl_dqn"

# Parse command line options
while [[ $# -gt 0 ]]; do
    case $1 in
        --episodes)      EPISODES="$2"; shift 2 ;;
        --batch-size)    BATCH_SIZE="$2"; shift 2 ;;
        --lr)            LR="$2"; shift 2 ;;
        --buffer-size)   BUFFER_SIZE="$2"; shift 2 ;;
        --epsilon-start) EPSILON_START="$2"; shift 2 ;;
        --epsilon-end)   EPSILON_END="$2"; shift 2 ;;
        --epsilon-decay) EPSILON_DECAY="$2"; shift 2 ;;
        --gamma)         GAMMA="$2"; shift 2 ;;
        --target-update) TARGET_UPDATE="$2"; shift 2 ;;
        --update-every)  UPDATE_EVERY="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: ./train_pong.sh [OPTIONS] [variants...]"
            echo ""
            echo "Options:"
            echo "  --episodes N        Number of episodes (default: $EPISODES)"
            echo "  --batch-size N      Batch size (default: $BATCH_SIZE)"
            echo "  --lr RATE           Learning rate (default: $LR)"
            echo "  --buffer-size N     Replay buffer size (default: $BUFFER_SIZE)"
            echo "  --epsilon-start N   Starting epsilon (default: $EPSILON_START)"
            echo "  --epsilon-end N     Final epsilon (default: $EPSILON_END)"
            echo "  --epsilon-decay N   Epsilon decay rate (default: $EPSILON_DECAY)"
            echo "  --gamma N           Discount factor (default: $GAMMA)"
            echo "  --target-update N   Target network update freq (default: $TARGET_UPDATE)"
            echo "  --update-every N    Network update frequency (default: $UPDATE_EVERY)"
            echo ""
            echo "Variants: dqn, ddqn, ddqn_duel, ddqn_per, ddqn_duel_per"
            exit 0
            ;;
        -*) echo "Unknown option: $1"; exit 1 ;;
        *) break ;;  # Start of variant names
    esac
done

# Function to get flags for a variant
get_flags() {
    case $1 in
        dqn)           echo "" ;;
        ddqn)          echo "--double" ;;
        ddqn_duel)     echo "--double --dueling" ;;
        ddqn_per)      echo "--double --priority" ;;
        ddqn_duel_per) echo "--double --dueling --priority" ;;
        *)             echo "INVALID" ;;
    esac
}

# Function to launch training in new Terminal
launch_training() {
    local variant=$1
    local flags=$(get_flags $variant)
    local save_dir="../output/pong/${variant}"

    if [[ "$flags" == "INVALID" ]]; then
        echo "WARNING: Unknown variant '$variant'"
        echo "Available: dqn, ddqn, ddqn_duel, ddqn_per, ddqn_duel_per"
        return 1
    fi

    echo "Launching: $variant"

    osascript <<EOF
tell application "Terminal"
    set newWindow to do script "cd ${CODE_DIR} && source ~/anaconda3/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} && python train.py --env PongNoFrameskip-v4 --episodes ${EPISODES} ${flags} --batch-size ${BATCH_SIZE} --lr ${LR} --buffer-size ${BUFFER_SIZE} --epsilon-start ${EPSILON_START} --epsilon-end ${EPSILON_END} --epsilon-decay ${EPSILON_DECAY} --gamma ${GAMMA} --target-update ${TARGET_UPDATE} --update-every ${UPDATE_EVERY} --save-dir ${save_dir}; exit"
    set bounds of front window to {100, 100, 1400, 600}
    set number of columns of front window to 180
    set number of rows of front window to 30
    activate
end tell
EOF
}

# Main
echo "=========================================="
echo "  Pong DQN Training Launcher"
echo "=========================================="
echo "  Episodes:      ${EPISODES}"
echo "  Batch size:    ${BATCH_SIZE}"
echo "  LR:            ${LR}"
echo "  Buffer:        ${BUFFER_SIZE}"
echo "  Epsilon:       ${EPSILON_START} -> ${EPSILON_END} (decay: ${EPSILON_DECAY})"
echo "  Gamma:         ${GAMMA}"
echo "  Target update: ${TARGET_UPDATE}"
echo "  Update every:  ${UPDATE_EVERY}"
echo "=========================================="

ALL_VARIANTS=(dqn ddqn ddqn_duel ddqn_per ddqn_duel_per)

# If no arguments, train all variants
if [[ $# -eq 0 ]]; then
    echo "Training ALL variants: ${ALL_VARIANTS[*]}"
    echo ""
    for variant in "${ALL_VARIANTS[@]}"; do
        launch_training "$variant"
        sleep 1  # Small delay between launches
    done
else
    # Train only specified variants
    echo "Training selected variants: $@"
    echo ""
    for variant in "$@"; do
        launch_training "$variant"
        sleep 1
    done
fi

echo ""
echo "All training sessions launched in separate Terminal windows."
echo "=========================================="
echo "Available variants:"
echo "  dqn           - Plain DQN"
echo "  ddqn          - Double DQN"
echo "  ddqn_duel     - Double + Dueling"
echo "  ddqn_per      - Double + PER"
echo "  ddqn_duel_per - Double + Dueling + PER"
echo "=========================================="
echo ""
echo "Once training completes, run this to generate plots:"
echo ""
echo "  cd ${CODE_DIR} && conda activate ${CONDA_ENV}"
echo ""
echo "  # Individual plots"
if [[ $# -eq 0 ]]; then
    for v in "${ALL_VARIANTS[@]}"; do
        echo "  python plot_metrics.py --dirs ../output/pong/${v} --save-dir ../output/pong/${v}"
    done
    echo ""
    echo "  # Comparison plot"
    echo "  python plot_metrics.py --dirs ../output/pong/dqn ../output/pong/ddqn ../output/pong/ddqn_duel ../output/pong/ddqn_per ../output/pong/ddqn_duel_per --save-dir ../output/pong/comparison"
else
    for v in "$@"; do
        echo "  python plot_metrics.py --dirs ../output/pong/${v} --save-dir ../output/pong/${v}"
    done
    if [[ $# -gt 1 ]]; then
        echo ""
        echo "  # Comparison plot"
        dirs=""
        for v in "$@"; do
            dirs="${dirs} ../output/pong/${v}"
        done
        echo "  python plot_metrics.py --dirs${dirs} --save-dir ../output/pong/comparison"
    fi
fi
echo "=========================================="
