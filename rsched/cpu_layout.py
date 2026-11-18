"""Split the CPUs this process may use into model-service cores and episode cores.

    python -m rsched.cpu_layout --service-cores 8 --episode-cores 24 --shell
    # SERVICE_CPUS=0,1,2,3,128,129,130,131 RS_INFRA_CPUS=... RS_EPISODE_CPUS=...

Every experiment that is compared with another must run under the same CPU
conditions; pods with the same GPU differ in CPU count and speed. With
``RS_PIN=1`` in env/.env, scripts/common.sh and rsched.sweep use this layout:
model servers, proxies and the GPU sampler on the service cores, episodes and
their simulators on the episode cores.

CPUs are taken in topology order (socket, physical core, hardware thread), so a
budget of k CPUs is k hardware threads on about k/2 physical cores when SMT is
on, and services and episodes never share a physical core. Logical CPU numbers
are not used directly: containers may expose any subset of the host's CPUs.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SYSFS = Path("/sys/devices/system/cpu")


def allowed_cpus() -> list[int]:
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return list(range(os.cpu_count() or 1))


def _read_int(path: Path, default: int) -> int:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return default


def ordered(cpus: list[int], sysfs: Path = SYSFS) -> list[int]:
    """CPUs sorted by (socket, physical core, logical id): SMT siblings end up adjacent."""
    def key(c: int) -> tuple[int, int, int]:
        topo = sysfs / f"cpu{c}" / "topology"
        return (_read_int(topo / "physical_package_id", 0), _read_int(topo / "core_id", c), c)
    return sorted(cpus, key=key)


def quota_cores(root: str = "/sys/fs/cgroup") -> float | None:
    """CPU time the cgroup may use, in cores (None = unlimited or unknown)."""
    for path, period in ((f"{root}/cpu.max", None), (f"{root}/cpu/cpu.cfs_quota_us", f"{root}/cpu/cpu.cfs_period_us"),
                         (f"{root}/cpu,cpuacct/cpu.cfs_quota_us", f"{root}/cpu,cpuacct/cpu.cfs_period_us")):
        try:
            text = Path(path).read_text().split()
        except OSError:
            continue
        if period is None:
            if text[0] == "max":
                return None
            return int(text[0]) / int(text[1])
        q = int(text[0])
        if q < 0:
            return None
        return q / int(Path(period).read_text().strip())
    return None


def layout(service_cores: int, episode_cores: int, cpus: list[int] | None = None,
           sysfs: Path = SYSFS) -> tuple[list[int], list[int]]:
    cpus = ordered(cpus if cpus is not None else allowed_cpus(), sysfs)
    need = service_cores + episode_cores
    if need > len(cpus):
        raise ValueError(f"need {need} CPUs ({service_cores} services + {episode_cores} episodes), "
                         f"only {len(cpus)} are available")
    return cpus[:service_cores], cpus[service_cores:need]


def fmt(cpus: list[int]) -> str:
    """Comma list with ranges, in the given order (taskset and rsched.sweep accept it)."""
    out, i = [], 0
    while i < len(cpus):
        j = i
        while j + 1 < len(cpus) and cpus[j + 1] == cpus[j] + 1:
            j += 1
        out.append(str(cpus[i]) if i == j else f"{cpus[i]}-{cpus[j]}")
        i = j + 1
    return ",".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--service-cores", type=int, default=8)
    ap.add_argument("--episode-cores", type=int, default=24)
    ap.add_argument("--shell", action="store_true", help="print shell assignments")
    ap.add_argument("--episodes-only", action="store_true", help="print only the episode CPU list")
    args = ap.parse_args(argv)
    try:
        svc, eps = layout(args.service_cores, args.episode_cores)
    except ValueError as exc:
        print(f"rsched.cpu_layout: {exc}", file=sys.stderr)
        return 1
    quota = quota_cores()
    if quota is not None and args.service_cores + args.episode_cores > quota + 1e-6:
        print(f"rsched.cpu_layout: WARNING {args.service_cores} + {args.episode_cores} CPUs exceed the cgroup "
              f"quota of {quota:g}; cells will be throttled", file=sys.stderr)
    if args.episodes_only:
        print(fmt(eps))
    elif args.shell:
        print(f"SERVICE_CPUS={fmt(svc)} RS_INFRA_CPUS={fmt(svc)} RS_EPISODE_CPUS={fmt(eps)}")
    else:
        print(f"service CPUs ({len(svc)}): {fmt(svc)}\nepisode CPUs ({len(eps)}): {fmt(eps)}\n"
              f"available: {len(allowed_cpus())}, cgroup quota: {quota if quota is not None else 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
