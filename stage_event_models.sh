#!/bin/bash
# Stage the hackathon's model servers on the serve reservation, one job (and
# one GPU) per model, or print their connection details once they're up.
#
#   ./stage_event_models.sh submit     # submit all four (can run the night before)
#   ./stage_event_models.sh status     # one Slack-ready block of endpoints + keys
#   ./stage_event_models.sh --dry-run submit
#
# Jobs submitted before the reservation opens pend with reason (Reservation)
# and start on their own at its start time. The window (34.5h) is shorter than
# the 48h cap, so each job's resubmit chain is switched off up front with
# STOP_CHAIN-<name>; see CLAUDE.md.
#
# Override any of these from the environment:
ACCOUNT="${ACCOUNT:-bfqi-delta-gpu}"
PARTITION="${PARTITION:-gpuA40x4}"
RESERVATION="${RESERVATION:-sup-30434}"
WALLTIME="${WALLTIME:-34:25:00}"
SERVE_DIR="${SERVE_DIR:-$HOME/.pathology/serve}"
# name=spec, one per server
MODELS="${MODELS:-conch=conch musk=musk quiltnet=openclip:wisdomik/QuiltNet-B-32 plip=hfclip:vinid/plip}"

# Load weights from the cache the smoke test used, offline: no surprise
# downloads at 07:30, and no dependence on HF_HOME from the submitting shell
# (sbatch exports the caller's environment to the job).
export HF_HOME="${EVENT_HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"   # run from anywhere; submit_serve.sbatch and logs/ live here

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then DRY_RUN=1; shift; fi
ACTION="${1:-}"

run() {
    if [ "$DRY_RUN" -eq 1 ]; then echo "DRY RUN: $*"; else "$@"; fi
}

submit() {
    mkdir -p "$SERVE_DIR" logs
    port="${BASE_PORT:-8000}"
    for entry in $MODELS; do
        name="${entry%%=*}"
        spec="${entry#*=}"
        run touch "$SERVE_DIR/STOP_CHAIN-$name"
        # Distinct starting port per server: jobs that start together on one
        # node would otherwise all pick the same free port.
        run sbatch --account="$ACCOUNT" --partition="$PARTITION" --reservation="$RESERVATION" \
            --time="$WALLTIME" --job-name="$name" --export="ALL,SERVE_PORT=$port" \
            submit_serve.sbatch "$spec"
        port=$((port + 1))
    done
}

status() {
    echo "Pathology model servers (call from inside Delta; same API for all):"
    for entry in $MODELS; do
        name="${entry%%=*}"
        spec="${entry#*=}"
        f="$SERVE_DIR/current-$name/connection.json"
        if [ -f "$f" ]; then
            python3 - "$f" "$name" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
print(f"- {sys.argv[2]}: http://{c['node']}:{c['port']}  key: {c['api_key']}")
PY
        else
            echo "- $name ($spec): not up yet"
        fi
    done
}

case "$ACTION" in
    submit) submit ;;
    status) status ;;
    *) echo "usage: $0 [--dry-run] submit|status" >&2; exit 2 ;;
esac
