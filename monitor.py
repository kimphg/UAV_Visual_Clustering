#!/usr/bin/env python3
"""Standalone system monitor for diagnosing silent training hangs/deaths.

Motivation: two remote training runs have died without ANY trace (no error,
no traceback, no OOM message, no coredump, no reboot) — confirmed via direct
server inspection both times. Neither dmesg nor journalctl is readable
without sudo on this box, so there is no way to find out what the kernel/GPU
driver saw. This gives an INDEPENDENT timeline (GPU state, memory, load,
disk, process liveness) that doesn't need elevated permissions, so a future
silent death can at least be bracketed: was the GPU still busy right before
it went quiet? Did free memory crater? Did the process vanish abruptly or
line up with something else?

Runs detached (same setsid+nohup+disown idiom as remote_train.py), polling
every INTERVAL seconds, appending one JSON line per sample to a log file —
cheap enough (~200-300 bytes/sample) to run continuously and not worth
rotating for a good while (871GB free on this server as of 2026-07-17).

Usage: python3 monitor.py [log_path] [--interval SECONDS] [--match PATTERN]
  log_path: defaults to monitor.log (relative to cwd, i.e. code_dir when
            launched the same way remote_train.py is)
  --match:  pgrep pattern for the training process to track liveness of
            (defaults to "remote_train.py")
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone


def nvidia_smi():
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,"
             "temperature.gpu,clocks.sm,clocks_event_reasons.active,"
             "ecc.errors.uncorrected.volatile.total",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5, stderr=subprocess.DEVNULL).strip()
        keys = ["gpu_util_pct", "mem_used_mib", "mem_total_mib", "power_w",
                "temp_c", "clock_sm_mhz", "throttle_reasons", "ecc_uncorrected"]
        vals = [v.strip() for v in out.split(",")]
        return dict(zip(keys, vals))
    except Exception as exc:
        return {"error": str(exc)}


def mem_info():
    try:
        d = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                d[k] = v.strip()
        return {
            "mem_total_kb": d.get("MemTotal", "").split()[0],
            "mem_available_kb": d.get("MemAvailable", d.get("MemFree", "0 kB")).split()[0],
            "swap_free_kb": d.get("SwapFree", "").split()[0],
        }
    except Exception as exc:
        return {"error": str(exc)}


def load_avg():
    try:
        parts = open("/proc/loadavg").read().split()
        return {"load1": parts[0], "load5": parts[1], "load15": parts[2]}
    except Exception as exc:
        return {"error": str(exc)}


def disk_free_gb():
    try:
        st = os.statvfs(os.path.expanduser("~"))
        return round(st.f_bavail * st.f_frsize / 1e9, 1)
    except Exception:
        return None


def training_process(match):
    """Liveness of the process matching `match` (e.g. "remote_train.py"),
    plus a coarse total python3 process count — DataLoader worker
    subprocesses spawned via multiprocessing do NOT contain the script name
    in their command line, so this can't count workers precisely; a sudden
    change in total python3 process count is still a useful coarse signal
    for "the worker pool collapsed" style failures."""
    try:
        out = subprocess.run(["pgrep", "-af", match],
                             capture_output=True, text=True).stdout.strip()
        pids = [l.split()[0] for l in out.splitlines() if l.strip()]
        n_py = subprocess.run(["pgrep", "-c", "python3"],
                              capture_output=True, text=True).stdout.strip()
        return {"alive": bool(pids), "pids": pids,
               "total_python3_procs": int(n_py) if n_py.isdigit() else None}
    except Exception as exc:
        return {"error": str(exc)}


def sample(match):
    return {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gpu": nvidia_smi(),
        "mem": mem_info(),
        "load": load_avg(),
        "disk_free_gb": disk_free_gb(),
        "training": training_process(match),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_path", nargs="?", default="monitor.log")
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--match", default="remote_train.py")
    args = ap.parse_args()

    with open(args.log_path, "a", buffering=1) as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "event": "monitor_started", "interval_s": args.interval}) + "\n")
        while True:
            f.write(json.dumps(sample(args.match)) + "\n")
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
