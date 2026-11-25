#!/usr/bin/env bash
# Report what is using the GPU before a load test, and fail if too little is free.
#
# nvidia-smi cannot enumerate processes under WSL2 -- --query-compute-apps comes
# back empty even while a job holds 9 GB. So consumers are found by scanning
# /proc/*/fd for open GPU device handles. Under WSL2 the GPU is exposed as
# /dev/dxg; /dev/nvidia* does not exist there, and scanning only for it finds
# nothing no matter what is running. Native Linux uses /dev/nvidia*, so both
# are matched.
#
# Each process is shown with its working directory and uptime, because the
# command line alone rarely says whose it is: the same virtualenv path appears
# for every project on the machine. Helper processes (compile workers,
# dataloader workers) are folded under the process that started them.
#
# Memory in use with no WSL process holding a handle is held Windows-side (the
# desktop compositor, a browser, another CUDA app). That memory is real and
# unavailable, but nothing inside WSL can free it, so this script reports it.
# It never stops anything: whether another process is needed is not a decision
# a script can make.
#
# Exit codes: 0 ok, 1 too little free memory, 2 cannot check, 3 strict mode and
# another process holds the GPU.
#
# Usage:  scripts/gpu_preflight.sh [min_free_mib]     (default 9000)
#         PREFLIGHT_STRICT=1 scripts/gpu_preflight.sh  # for timing measurements
#
# Strict mode matters for anything whose speed will be quoted. A second process
# can slow a run by taking compute while leaving plenty of memory free, which
# the free-memory check cannot see.
set -uo pipefail

MIN_FREE_MIB="${1:-9000}"
STRICT="${PREFLIGHT_STRICT:-0}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "gpu-preflight: nvidia-smi not found; cannot verify GPU state" >&2
    exit 2
fi

read -r USED TOTAL UTIL <<<"$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits | tr -d ',')"
FREE=$(( TOTAL - USED ))

echo "GPU: ${USED} MiB used / ${TOTAL} MiB total (${FREE} MiB free, ${UTIL}% util)"

# --- consumers visible inside WSL ---
HOLDERS=()
for dir in /proc/[0-9]*; do
    pid=${dir#/proc/}
    if ls -l "$dir/fd" 2>/dev/null | grep -qE "/dev/(nvidia|dxg)"; then
        HOLDERS+=("$pid")
    fi
done

declare -A IS_HOLDER=()
for pid in ${HOLDERS[@]+"${HOLDERS[@]}"}; do IS_HOLDER[$pid]=1; done

# Walk up through parents that also hold the GPU to find the process that
# started the group.
root_of() {
    local p=$1 parent
    while :; do
        parent=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
        if [ -n "$parent" ] && [ -n "${IS_HOLDER[$parent]:-}" ]; then p=$parent; else break; fi
    done
    echo "$p"
}

ROOTS=()
declare -A HELPERS=()
for pid in ${HOLDERS[@]+"${HOLDERS[@]}"}; do
    root=$(root_of "$pid")
    if [ "$root" = "$pid" ]; then
        ROOTS+=("$pid")
    else
        HELPERS[$root]=$(( ${HELPERS[$root]:-0} + 1 ))
    fi
done

if [ ${#ROOTS[@]} -gt 0 ]; then
    echo
    echo "Processes holding the GPU inside WSL:"
    for pid in "${ROOTS[@]}"; do
        printf "  pid %-8s up %-10s %s\n" "$pid" \
            "$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')" \
            "$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-90)"
        printf "      project: %s" "$(readlink "/proc/$pid/cwd" 2>/dev/null)"
        if [ -n "${HELPERS[$pid]:-}" ]; then printf "   (+%s helper processes)" "${HELPERS[$pid]}"; fi
        echo
    done
    echo
    echo "  Whether these are needed is your decision; this script stops nothing."
    echo "  To stop one:  kill <pid>"
else
    echo "  no GPU processes inside WSL"
fi

# --- memory unaccounted for by WSL processes ---
if [ ${#ROOTS[@]} -eq 0 ] && [ "$USED" -gt 500 ]; then
    echo
    echo "NOTE: ${USED} MiB is in use but no process in WSL holds a GPU handle."
    echo "      That memory belongs to the Windows side (desktop compositor, a"
    echo "      browser with hardware acceleration, another CUDA app). Nothing"
    echo "      in WSL can release it."
fi

if [ "$FREE" -lt "$MIN_FREE_MIB" ]; then
    echo
    echo "FAIL: ${FREE} MiB free, need ${MIN_FREE_MIB} MiB."
    echo "      Training on a 10 GB card has no headroom to spare; the run will"
    echo "      thrash the paged optimizer rather than fail cleanly, which looks"
    echo "      like 100% GPU utilization with the step counter frozen."
    exit 1
fi

if [ "$STRICT" = "1" ] && [ ${#ROOTS[@]} -gt 0 ]; then
    echo
    echo "FAIL (strict): another process holds the GPU, so timings from this run"
    echo "      would include its load. Stop it, or run without PREFLIGHT_STRICT"
    echo "      if you will not quote the speed."
    exit 3
fi

echo
echo "OK: ${FREE} MiB free (>= ${MIN_FREE_MIB} MiB required)"
