"""Tests for the Windows memory watchdog's decisions (no WSL, any OS): python tests/test_memory_watchdog.py"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "windows-memory-watchdog"))
import memory_watchdog as mw  # noqa: E402

RESULTS = []


def check(cond, msg):
    RESULTS.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + msg)


MEMINFO = """MemTotal:       22543348 kB
MemFree:         1769468 kB
MemAvailable:   17998236 kB
Buffers:          524288 kB
Cached:         17825792 kB
SwapCached:         1024 kB
Shmem:            327680 kB
HugePages_Total:       0
"""

cache = mw.parse_meminfo(MEMINFO)
check(abs(cache - (524288 + 17825792 - 327680) / 1024 ** 2) < 1e-9, f"cache = Buffers + Cached - Shmem ({cache:.2f} GB)")
check(mw.parse_meminfo("") == 0, "empty meminfo reads as no cache")

cfg = dict(mw.DEFAULTS)
check(mw.should_drop(1.3, 17.3, cfg), "drops when Windows is low and the VM holds lots of cache")
check(not mw.should_drop(13.0, 17.3, cfg), "leaves the cache alone while Windows has room")
check(not mw.should_drop(1.3, 0.4, cfg), "does nothing when the VM has little cache to give back")
check(not mw.should_drop(4.0, 17.3, cfg), "the threshold itself is not low")
check(mw.should_drop(5.0, 3.0, dict(cfg, free_below_gb=6, min_cache_gb=3)), "config.json values replace the defaults")

print(f"{sum(RESULTS)}/{len(RESULTS)} passed")
sys.exit(0 if all(RESULTS) else 1)
