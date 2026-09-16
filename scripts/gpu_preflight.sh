#!/usr/bin/env bash
# Report what is using the GPU before a load test, and fail if too little is free.
#
# nvidia-smi cannot enumerate processes under WSL2 -- --query-compute-apps comes
# back empty even while a job holds 9 GB. So consumers are found by scanning
# /proc for open /dev/nvidia* handles, which does work inside the WSL namespace.
#
# Memory in use with no WSL process holding a handle is held Windows-side (the
# desktop compositor, a browser, another CUDA app). That memory is real and
# unavailable, but nothing inside WSL can free it -- it needs a decision on the
# Windows side, so this script reports it rather than trying to act.
#
# Usage:  scripts/gpu_preflight.sh [min_free_mib]     (default 9000)
set -uo pipefail

MIN_FREE_MIB="${1:-9000}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "gpu-preflight: nvidia-smi not found; cannot verify GPU state" >&2
    exit 2
fi

read -r USED TOTAL UTIL <<<"$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits | tr -d ',')"
FREE=$(( TOTAL - USED ))

echo "GPU: ${USED} MiB used / ${TOTAL} MiB total (${FREE} MiB free, ${UTIL}% util)"

# --- consumers visible inside WSL ---
WSL_PIDS=()
for pid in /proc/[0-9]*; do
    pid=${pid#/proc/}
    if ls -l "/proc/$pid/fd" 2>/dev/null | grep -q "/dev/nvidia"; then
        WSL_PIDS+=("$pid")
    fi
done

if [ ${#WSL_PIDS[@]} -gt 0 ]; then
    echo
    echo "Processes holding the GPU inside WSL:"
    for pid in "${WSL_PIDS[@]}"; do
        printf "  pid %-8s %-16s %s\n" "$pid" \
            "$(cat "/proc/$pid/comm" 2>/dev/null)" \
            "$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-72)"
    done
    echo
    echo "  -> these are yours to stop:  kill <pid>"
else
    echo "  no GPU processes inside WSL"
fi

# --- memory unaccounted for by WSL processes ---
if [ ${#WSL_PIDS[@]} -eq 0 ] && [ "$USED" -gt 500 ]; then
    echo
    echo "NOTE: ${USED} MiB is in use but no WSL process holds a GPU handle."
    echo "      That memory belongs to the Windows side (desktop compositor, a"
    echo "      browser with hardware acceleration, another CUDA app). Nothing"
    echo "      in WSL can release it. Closing those applications is a decision"
    echo "      for you, not something this script should do."
fi

if [ "$FREE" -lt "$MIN_FREE_MIB" ]; then
    echo
    echo "FAIL: ${FREE} MiB free, need ${MIN_FREE_MIB} MiB."
    echo "      Training on a 10 GB card has no headroom to spare; the run will"
    echo "      thrash the paged optimizer rather than fail cleanly, which looks"
    echo "      like 100% GPU utilization with the step counter frozen."
    exit 1
fi

echo
echo "OK: ${FREE} MiB free (>= ${MIN_FREE_MIB} MiB required)"
