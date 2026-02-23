#!/usr/bin/env bash
# Connect to tmux: attach if exactly one session, otherwise ask user to choose.

set -e

list_sessions() {
  tmux list-sessions 2>/dev/null || true
}

session_count() {
  list_sessions | wc -l
}

count=$(session_count)

if [[ "$count" -eq 0 ]]; then
  echo "No tmux sessions found. Start one with: tmux new -s <name>"
  exit 1
fi

if [[ "$count" -eq 1 ]]; then
  session=$(list_sessions | head -1 | sed -n 's/^\([^:]*\):.*/\1/p')
  exec tmux attach-session -t "$session"
fi

# Multiple sessions: show menu and let user choose
echo "Multiple tmux sessions:"
echo ""
n=1
while IFS= read -r line; do
  name=$(echo "$line" | sed -n 's/^\([^:]*\):.*/\1/p')
  echo "  $n) $name"
  n=$((n + 1))
done < <(list_sessions)
echo "  q) Quit"
echo ""

read -r -p "Select session [1-$((n - 1))]: " choice

if [[ "$choice" == "q" || "$choice" == "Q" || -z "$choice" ]]; then
  exit 0
fi

if ! [[ "$choice" =~ ^[0-9]+$ ]] || [[ "$choice" -lt 1 ]] || [[ "$choice" -gt $((n - 1)) ]]; then
  echo "Invalid choice."
  exit 1
fi

session=$(list_sessions | sed -n "${choice}p" | sed -n 's/^\([^:]*\):.*/\1/p')
exec tmux attach-session -t "$session"
