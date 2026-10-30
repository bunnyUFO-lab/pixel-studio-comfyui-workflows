"""Tests for the Windows memory watchdog's decisions and VRAM view, and the ComfyUI route it reads (no WSL, GPU or
ComfyUI needed, any OS): python tests/test_memory_watchdog.py"""
import importlib.util
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

check(mw.idle_drop_due(10, 6.1, cfg), "drops what a past job left once ComfyUI has been idle 10 min")
check(not mw.idle_drop_due(9.9, 6.1, cfg), "waits out the idle time first")
check(not mw.idle_drop_due(60, 1.1, cfg), "an idle VM with little cache is left alone")
check(not mw.idle_drop_due(600, 20, dict(cfg, idle_minutes=0)), "idle_minutes 0 turns the idle drop off")

mw.HOME = os.path.join(ROOT, "tests", ".watchdog-state")      # State writes state.json under HOME
os.makedirs(mw.HOME, exist_ok=True)
try:
    state = mw.State()
    state.set("comfy-busy", 123.0)
    check(mw.State().last["comfy-busy"] == 123.0, "state survives between runs")
    check(state.due("idle-check", 5) and not state.due("idle-check", 5), "a check is due once per its interval")
finally:
    os.remove(os.path.join(mw.HOME, "state.json"))
    os.rmdir(mw.HOME)

groups, apps = mw.group_processes([("vmwp", 15.6), ("dwm", 1.7), ("chrome", 0.4), ("chrome", 0.2), ("ollama", 3.0),
                                    ("explorer", 0.2)])
check(groups == {"ComfyUI's VM": 15.6, "Ollama": 3.0, "Windows desktop": 1.7}, "VM, Ollama and desktop get their own lines")
check([a for a, _ in apps] == ["chrome", "explorer"] and abs(apps[0][1] - 0.6) < 1e-9, "other apps add up per name, biggest first")

report = {"gpu": "NVIDIA GeForce RTX 4090", "total": 24.0, "free": 1.2, "groups": groups, "apps": apps,
          "comfy": [("WAN21_Vace", 13.9, 16.2), ("WanTEModel", 1.2, 10.6)], "ollama": [("qwen3.5:9b", 3.0)]}
line = mw.vram_line(report)
check(line.startswith("low VRAM: 1.2 of 24.0 GB free on NVIDIA GeForce RTX 4090"), "the log line leads with free VRAM")
check("ComfyUI's VM 15.6 GB: WAN21_Vace 13.9, WanTEModel 1.2" in line and "Ollama 3.0 GB: qwen3.5:9b 3.0" in line,
      "the log line names ComfyUI's and Ollama's models")
status = "\n".join(mw.vram_status(report, cfg))
check("22.8 of 24.0 GB used" in status and "WAN21_Vace 13.9 of 16.2" in status, "--status shows use and each model's share")
check("Ollama: not running" in "\n".join(mw.vram_status(dict(report, ollama="not running"), cfg)), "Ollama off is said plainly")
check("no NVIDIA GPU" in mw.vram_status(None, cfg)[0], "no GPU: one line, no crash")

# ComfyUI side: the route's model list, from stand-ins for comfy.model_management.LoadedModel
spec = importlib.util.spec_from_file_location("memory_route", os.path.join(
    ROOT, "custom_nodes", "comfyui_workflows_utils", "memory.py"))
route = importlib.util.module_from_spec(spec)
spec.loader.exec_module(route)                   # outside ComfyUI: the route is skipped, the function stays


class FakeLoaded:
    def __init__(self, cls_name, loaded, size, used=False, gone=False):
        self.model = None if gone else type("Patcher", (), {"model": type(cls_name, (), {})()})()
        self.device, self.currently_used, self._loaded, self._size = "cuda:0", used, loaded, size

    def model_loaded_memory(self):
        return self._loaded

    def model_memory(self):
        return self._size


MB = 1024 * 1024
models = route.loaded_models([FakeLoaded("WanVAE", 240 * MB, 240 * MB), FakeLoaded("WAN21_Vace", 14000 * MB, 16542 * MB, True),
                              FakeLoaded("Gone", 1, 1, gone=True)])
check([m["name"] for m in models] == ["WAN21_Vace", "WanVAE"], "the route lists live models by VRAM share, skips gone ones")
check(models[0] == {"name": "WAN21_Vace", "device": "cuda:0", "vram_mb": 14000, "size_mb": 16542, "in_use": True},
      "each model has its VRAM share, full size and whether a job uses it")

print(f"{sum(RESULTS)}/{len(RESULTS)} passed")
sys.exit(0 if all(RESULTS) else 1)
