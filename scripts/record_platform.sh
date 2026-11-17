#!/usr/bin/env bash
# Write what identifies this machine to <out>/platform.txt: host, GPU, driver,
# clocks and power limit, CPU model, cgroup CPU quota, memory. Pods with the same
# GPU differ by up to ~1.8x in simulator step time and ~1.5x in Pi0.5 latency, so
# every result directory should say which machine it came from.
#
#   bash scripts/record_platform.sh results/<experiment>/<stamp>
set -uo pipefail
out="${1:?usage: $0 <result dir>}"
mkdir -p "$out"
{
  echo "date: $(date -Is)"
  echo "host: $(hostname)"
  echo "cgroup cpu.max: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo "$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null) $(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null)")"
  echo "nproc: $(nproc)"
  lscpu 2>/dev/null | grep -E "Model name|^CPU\(s\)|MHz" | sed 's/^/cpu: /'
  free -g 2>/dev/null | awk '/^Mem:/ {print "memory GB: " $2}'
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.limit,pcie.link.gen.current \
    --format=csv 2>/dev/null | sed 's/^/gpu: /'
  echo "mps: $(echo get_server_list | nvidia-cuda-mps-control >/dev/null 2>&1 && echo on || echo off)"
} > "$out/platform.txt"
echo "platform -> $out/platform.txt"
