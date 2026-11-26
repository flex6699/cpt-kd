#!/usr/bin/env bash
# Start the first KD vs CPT-KD test inside tmux: one window per GPU, everything logged.
#
# Usage (from ~/ml_research):
#   bash run_first_test.sh
# Then:
#   tmux attach -t teakd        # watch;  Ctrl-b then d  to leave it running
#   tail -f logs/*.log          # or just follow the log files
set -euo pipefail

SESSION="teakd"
PROJECT="$HOME/ml_research"
ENV="teakd"
DATA="data/tea_kd/cub"

# commands for each GPU, run one after another
GPU1_CMDS=(
  "python train_student.py --data $DATA --method kd  --gpu 1"
  "python train_student.py --data $DATA --method ce  --gpu 1"
)
GPU2_CMDS=(
  "python train_student.py --data $DATA --method cpt --gpu 2"
  "python train_student.py --data $DATA --method pergate --teacher-source oof --gpu 2"
)

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "A tmux session called '$SESSION' already exists."
  echo "Attach with: tmux attach -t $SESSION   (or remove it with: tmux kill-session -t $SESSION)"
  exit 1
fi
mkdir -p "$PROJECT/logs"

# build one shell line: activate env, go to project, run each command with its own log file
chain() {
  local line="conda activate $ENV && cd $PROJECT"
  for cmd in "$@"; do
    local method
    method=$(echo "$cmd" | sed -E 's/.*--method ([a-z_]+).*/\1/')
    local extra=""
    [[ "$cmd" == *"--teacher-source oof"* ]] && extra="_oofteacher"
    line+=" && $cmd 2>&1 | tee logs/cub_${method}${extra}_s0.log"
  done
  line+=" ; echo; echo '=== all runs in this window finished ==='"
  echo "$line"
}

tmux new-session -d -s "$SESSION" -n gpu1
tmux send-keys -t "$SESSION:gpu1" "$(chain "${GPU1_CMDS[@]}")" C-m
tmux new-window -t "$SESSION" -n gpu2
tmux send-keys -t "$SESSION:gpu2" "$(chain "${GPU2_CMDS[@]}")" C-m

echo "Started tmux session '$SESSION' with windows gpu1 and gpu2."
echo "  Watch:          tmux attach -t $SESSION"
echo "  Switch window:  Ctrl-b then n   (next)   /  Ctrl-b then p (previous)"
echo "  Leave running:  Ctrl-b then d"
echo "  Logs:           $PROJECT/logs/"
echo "  Quick summary:  grep -h 'top-1' $PROJECT/logs/cub_*.log"
