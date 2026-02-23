#!/bin/bash
# Wrapper script for run that automatically activates the virtual environment
# and runs inside a tmux session when not already in one

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# If not in tmux, create a session and run there (then attach)
if [ -z "${TMUX:-}" ]; then
    SESSION_BASE="run_session"
    SESSION_NAME="$SESSION_BASE"
    SESSION_INDEX=1

    # Ensure we do not fail if the base session already exists.
    while tmux has-session -t "$SESSION_NAME" 2>/dev/null; do
        SESSION_NAME="${SESSION_BASE}_${SESSION_INDEX}"
        SESSION_INDEX=$((SESSION_INDEX + 1))
    done

    exec tmux new-session -s "$SESSION_NAME" bash -c "cd '$SCRIPT_DIR' && '$SCRIPT_DIR/run.sh' \"\$@\"; exec bash" _ "$@"
fi

# Check if venv exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Virtual environment not found at $VENV_DIR" >&2
    echo "Please create a virtual environment first." >&2
    exit 1
fi

# Activate the virtual environment and run the installed run command
source "$VENV_DIR/bin/activate"
exec "$VENV_DIR/bin/run" "$@"

