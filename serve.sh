#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 || $# > 4 )); then
    printf 'Usage: %s PATH_TO_NINFER_SERVE MODEL.ninfer [PORT] [none|dspark]\n' "$0" >&2
    exit 2
fi

port=${3:-8080}
spec=${4:-none}
if [[ ! $port =~ ^[1-9][0-9]{0,4}$ ]] || (( port > 65535 )); then
    printf 'Invalid port: %s\n' "$port" >&2
    exit 2
fi
if [[ $spec != none && $spec != dspark ]]; then
    printf 'Invalid speculation mode: %s\n' "$spec" >&2
    exit 2
fi

exec "$1" "$2" --host 127.0.0.1 --port "$port" --model-id inkling-2-mini \
    --max-context 16384 --max-concurrency 1 --expert-residency device \
    --kv-dtype fp16 --spec "$spec"
