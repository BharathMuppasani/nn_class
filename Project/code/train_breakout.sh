#!/bin/zsh
# Train Breakout DQN variants - each in its own Terminal window
# Usage: ./train_breakout.sh [variants...]
# Examples:
#   ./train_breakout.sh                    # Train all 5 variants
#   ./train_breakout.sh dqn ddqn           # Train only DQN and Double DQN
#   ./train_breakout.sh ddqn_duel_per      # Train only Double+Dueling+PER

CODE_DIR="/Users/bittu/Desktop/GitHub/nn_class/Project/code"
CONDA_ENV="rl_dqn"
EPISODES=3000
BATCH_SIZE=128
LR="3e-4"
BUFFER_SIZE=100000

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
    local save_dir="../output/breakout/${variant}"

    if [[ "$flags" == "INVALID" ]]; then
        echo "WARNING: Unknown variant '$variant'"
        echo "Available: dqn, ddqn, ddqn_duel, ddqn_per, ddqn_duel_per"
        return 1
    fi

    echo "Launching: $variant"

    osascript <<EOF
tell application "Terminal"
    do script "cd ${CODE_DIR} && source ~/anaconda3/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} && python train.py --env BreakoutNoFrameskip-v4 --episodes ${EPISODES} ${flags} --batch-size ${BATCH_SIZE} --lr ${LR} --buffer-size ${BUFFER_SIZE} --save-dir ${save_dir}; exit"
    activate
end tell
EOF
}

# Main
echo "=========================================="
echo "  Breakout DQN Training Launcher"
echo "=========================================="
echo "  Episodes:    ${EPISODES}"
echo "  Batch size:  ${BATCH_SIZE}"
echo "  LR:          ${LR}"
echo "  Buffer:      ${BUFFER_SIZE}"
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
        echo "  python plot_metrics.py --dirs ../output/breakout/${v} --save-dir ../output/breakout/${v}"
    done
    echo ""
    echo "  # Comparison plot"
    echo "  python plot_metrics.py --dirs ../output/breakout/dqn ../output/breakout/ddqn ../output/breakout/ddqn_duel ../output/breakout/ddqn_per ../output/breakout/ddqn_duel_per --save-dir ../output/breakout/comparison"
else
    for v in "$@"; do
        echo "  python plot_metrics.py --dirs ../output/breakout/${v} --save-dir ../output/breakout/${v}"
    done
    if [[ $# -gt 1 ]]; then
        echo ""
        echo "  # Comparison plot"
        dirs=""
        for v in "$@"; do
            dirs="${dirs} ../output/breakout/${v}"
        done
        echo "  python plot_metrics.py --dirs${dirs} --save-dir ../output/breakout/comparison"
    fi
fi
echo "=========================================="
