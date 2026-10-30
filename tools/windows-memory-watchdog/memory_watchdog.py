"""Hands the Docker VM's file cache back to Windows when Windows runs short of RAM.

ComfyUI in Docker Desktop runs inside a WSL2 VM. Loading a big model (Wan 2.x stages ~14 GB) fills
the VM with file cache, Windows drops below 1 GB free and the job sits at "Model Initializing" for
minutes. WSL's own autoMemoryReclaim only reclaims once the VM is idle. This script, started every
minute by the "ComfyUI memory watchdog" scheduled task (install.ps1), looks every 10 seconds and
drops that cache as soon as Windows' available RAM falls below a threshold and the VM holds enough
cache to matter:

    wsl -d docker-desktop -u root sh -c "sync; echo 1 > /proc/sys/vm/drop_caches"

It also tidies up when things go quiet: once ComfyUI has had no job queued or running for idle_minutes
(Sprite Studio has unloaded its models by then) and the VM still holds idle_cache_gb or more of cache, it
drops that cache too, so the RAM a past job (or the boot) left behind goes back to Windows without waiting
for memory to run short. It looks at most every IDLE_CHECK_MINUTES and only reads ComfyUI's GET /queue
before that, so an idle PC costs one local HTTP call per check.

Only clean cache goes; files ComfyUI is using stay mapped. It never stops ComfyUI, Docker or WSL,
and it does not start the VM when it is off. The task runs it with pythonw.exe and its children
with CREATE_NO_WINDOW, so no window ever opens or takes focus.

It also keeps an eye on the GPU, read-only: when its free memory falls below vram_below_gb it logs who
holds the VRAM (ComfyUI's VM and the models ComfyUI has loaded, Ollama and its models, the Windows
desktop, other apps), at most once per QUIET_MINUTES. It never frees VRAM itself.

Settings: config.json beside the installed copy (keys as in DEFAULTS) overrides the defaults.
Log: %LOCALAPPDATA%\\memory-watchdog\\watchdog.log, one line per action, rotated at 256 KB.

    python memory_watchdog.py --status   # print RAM, VM cache and VRAM and what it would do; change nothing
"""
import ctypes
import json
import logging
import logging.handlers
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

DEFAULTS = {
    "free_below_gb": 4.0,   # act when Windows' available RAM is below this
    "min_cache_gb": 2.0,    # ...and the VM holds at least this much droppable file cache
    "idle_minutes": 10,     # also drop once ComfyUI has had no job for this long (0 = only when memory is low)
    "idle_cache_gb": 3.0,   # ...and the VM holds at least this much cache
    "distro": "docker-desktop",
    "vram_below_gb": 1.5,   # log who holds the GPU's memory when less than this is free
    "comfy_url": "http://127.0.0.1:8188",
    "ollama_url": "http://127.0.0.1:11434",
}
HOME = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "memory-watchdog")
CHECK_SECONDS = 10          # the task starts this every minute; it checks this often until the next start
RUN_SECONDS = 50
QUIET_MINUTES = 30          # repeat a "low but nothing to drop" or error line at most this often
IDLE_CHECK_MINUTES = 5      # while ComfyUI is idle, read the VM's cache at most this often
GIB = 1024 ** 3


def parse_meminfo(text):
    """Droppable file cache in GiB from /proc/meminfo: Buffers + Cached minus Shmem (tmpfs/shm, not droppable)."""
    kb = {}
    for line in text.splitlines():
        name, _, rest = line.partition(":")
        if rest.split():
            kb[name.strip()] = int(rest.split()[0])
    return (kb.get("Buffers", 0) + kb.get("Cached", 0) - kb.get("Shmem", 0)) / 1024 ** 2


def should_drop(free_gb, cache_gb, cfg):
    return free_gb < cfg["free_below_gb"] and cache_gb >= cfg["min_cache_gb"]


def idle_drop_due(idle_minutes, cache_gb, cfg):
    return 0 < cfg["idle_minutes"] <= idle_minutes and cache_gb >= cfg["idle_cache_gb"]


def windows_available_gb():
    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile",
                                                    "ullAvailPageFile", "ullTotalVirtual", "ullAvailVirtual",
                                                    "ullAvailExtendedVirtual")]
    status = MemoryStatusEx(dwLength=ctypes.sizeof(MemoryStatusEx))
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return status.ullAvailPhys / GIB


def wsl(*args, timeout=60):
    """Run wsl.exe without a console window and return its output (wsl.exe may answer in UTF-16)."""
    result = subprocess.run(["wsl.exe", *args], stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout,
                            creationflags=subprocess.CREATE_NO_WINDOW, env=dict(os.environ, WSL_UTF8="1"))
    out = result.stdout.decode("utf-8", "replace").replace("\x00", "")
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", "replace").replace("\x00", "").strip()
        raise RuntimeError(f"wsl {' '.join(args[:3])} exited {result.returncode}: {err or out.strip()}"[:300])
    return out


def vm_running(distro):
    try:
        return distro in wsl("--list", "--running", "--quiet", timeout=20).split()
    except RuntimeError:
        return False   # "There are no running distributions" exits non-zero


def vm_cache_gb(distro):
    return parse_meminfo(wsl("-d", distro, "-u", "root", "sh", "-c", "cat /proc/meminfo"))


# ----------------------------------------------------------------- VRAM: who holds the GPU's memory (read-only)
VRAM_GROUPS = (                       # process names (lower case, no .exe) -> the line they are shown on
    ("ComfyUI's VM", ("vmwp", "vmmem", "vmmemwsl")),     # Docker Desktop's WSL VM; Windows sees its GPU memory here
    ("Ollama", ("ollama", "ollama app", "llama-server", "ollama_llama_server")),
    ("Windows desktop", ("dwm",)),
)
PROCESS_VRAM = (r"$by = @{}; foreach ($s in (Get-Counter '\GPU Process Memory(*)\Dedicated Usage' -ErrorAction Stop)"
                r".CounterSamples) { if ($s.InstanceName -match '^pid_(\d+)_') { $by[[int]$matches[1]] += $s.CookedValue } };"
                r" ConvertTo-Json -Compress -InputObject @($by.GetEnumerator() | Where-Object Value -gt 0 | ForEach-Object {"
                r" @{ name = (Get-Process -Id $_.Key -ErrorAction SilentlyContinue).ProcessName; bytes = [long]$_.Value } })")


class NvmlMemory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


_nvml = None                          # nvml.dll once loaded; False when there is none (no NVIDIA driver)


def gpu_memory():
    """(name, total_gb, free_gb) of GPU 0 from NVML, which comes with the NVIDIA driver; None without one."""
    global _nvml
    if _nvml is None:
        try:
            lib = ctypes.WinDLL("nvml.dll")
            _nvml = lib if lib.nvmlInit_v2() == 0 else False
        except (OSError, AttributeError):
            _nvml = False
    if not _nvml:
        return None
    handle, mem, name = ctypes.c_void_p(), NvmlMemory(), ctypes.create_string_buffer(96)
    if _nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) or _nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem)):
        return None
    _nvml.nvmlDeviceGetName(handle, name, 96)
    return name.value.decode("utf-8", "replace"), mem.total / GIB, mem.free / GIB


def process_vram():
    """[(process name, gb)] of each process's dedicated GPU memory, from Windows' GPU counters ([] when unreadable,
    e.g. on a non-English Windows, whose counter names differ). Takes about 2 s, so only --status and the low-VRAM
    log call it."""
    try:
        result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", PROCESS_VRAM],
                                stdin=subprocess.DEVNULL, capture_output=True, timeout=30,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        return [(p.get("name") or "?", p["bytes"] / GIB) for p in json.loads(result.stdout or b"[]")]
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired):
        return []


def group_processes(procs):
    """[(name, gb)] -> ({group: gb} for VRAM_GROUPS, [(app, gb)] for everything else, biggest first)."""
    groups, apps = {label: 0.0 for label, _ in VRAM_GROUPS}, {}
    for name, gb in procs:
        label = next((label for label, names in VRAM_GROUPS if name.lower() in names), None)
        if label:
            groups[label] += gb
        else:
            apps[name] = apps.get(name, 0.0) + gb
    return groups, sorted(apps.items(), key=lambda a: -a[1])


def http_json(url, timeout=3):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def comfy_models(cfg):
    """[(name, vram_gb, size_gb)] ComfyUI holds now, or a short reason why it cannot say. The list comes from
    GET /spritestudio/loaded_models (custom_nodes/comfyui_workflows_utils/memory.py in this repository)."""
    try:
        models = http_json(cfg["comfy_url"].rstrip("/") + "/spritestudio/loaded_models")["models"]
        return [(m["name"], m["vram_mb"] / 1024, m["size_mb"] / 1024) for m in models]
    except urllib.error.HTTPError as e:
        return "its model list needs this repository's comfyui_workflows_utils node" if e.code == 404 else f"HTTP {e.code}"
    except (OSError, ValueError, KeyError, TypeError):
        return "not answering"


def comfy_busy(cfg):
    """True while ComfyUI has a job running or queued. Not answering counts as idle: it has no job then either."""
    try:
        queue = http_json(cfg["comfy_url"].rstrip("/") + "/queue")
        return bool(queue.get("queue_running") or queue.get("queue_pending"))
    except (OSError, ValueError, AttributeError):
        return False


def ollama_models(cfg):
    """[(name, vram_gb)] Ollama has loaded now, or "not running"."""
    try:
        return [(m["name"], m.get("size_vram", 0) / GIB) for m in http_json(cfg["ollama_url"].rstrip("/") + "/api/ps")["models"]]
    except (OSError, ValueError, KeyError, TypeError):
        return "not running"


def vram_report(cfg, gpu=None):
    """Everything --status and the low-VRAM log show, or None without an NVIDIA GPU."""
    gpu = gpu or gpu_memory()
    if gpu is None:
        return None
    groups, apps = group_processes(process_vram())
    return {"gpu": gpu[0], "total": gpu[1], "free": gpu[2], "groups": groups, "apps": apps,
            "comfy": comfy_models(cfg), "ollama": ollama_models(cfg)}


def _models(models, with_size=False):
    if isinstance(models, str):
        return models
    if not models:
        return "no models loaded"
    return ", ".join(f"{m[0]} {m[1]:.1f}" + (f" of {m[2]:.1f}" if with_size else "") for m in models)


def _apps(apps, top=4):
    shown = [f"{name} {gb:.1f}" for name, gb in apps[:top]]
    rest = sum(gb for _, gb in apps[top:])
    return ", ".join(shown + ([f"others {rest:.1f}"] if rest >= 0.05 else []))


def vram_line(r):
    """One log line: free VRAM, then each holder with what it holds."""
    g = r["groups"]
    return (f"low VRAM: {r['free']:.1f} of {r['total']:.1f} GB free on {r['gpu']} | "
            f"ComfyUI's VM {g[VRAM_GROUPS[0][0]]:.1f} GB: {_models(r['comfy'])} | "
            f"Ollama {g['Ollama']:.1f} GB: {_models(r['ollama'])} | Windows desktop {g['Windows desktop']:.1f} GB | "
            f"apps {sum(gb for _, gb in r['apps']):.1f} GB: {_apps(r['apps'])}")


def vram_status(r, cfg):
    """--status lines, GB throughout."""
    if r is None:
        return ["GPU: no NVIDIA GPU found (nvml.dll), so no VRAM view"]
    g = r["groups"]
    lines = [f"GPU {r['gpu']}: {r['total'] - r['free']:.1f} of {r['total']:.1f} GB used, {r['free']:.1f} GB free "
             f"(logs who holds it below {cfg['vram_below_gb']} GB free)",
             f"  {g[VRAM_GROUPS[0][0]]:5.1f}  ComfyUI's VM (Docker/WSL): {_models(r['comfy'], with_size=True)}",
             f"  {g['Ollama']:5.1f}  Ollama: {_models(r['ollama'])}",
             f"  {g['Windows desktop']:5.1f}  Windows desktop (dwm)",
             f"  {sum(gb for _, gb in r['apps']):5.1f}  apps: {_apps(r['apps']) or 'none'}"]
    if not any(g.values()) and not r["apps"]:
        lines.append("  (Windows' GPU counters were not readable, so the split above is missing)")
    return lines


def check_vram(cfg, log):
    gpu = gpu_memory()
    if gpu is None or gpu[2] >= cfg["vram_below_gb"]:
        return                                   # the usual case: one NVML call, no log line
    if State().due("low-vram"):
        log.info(vram_line(vram_report(cfg, gpu)))


def load_config():
    cfg = dict(DEFAULTS)
    try:
        with open(os.path.join(HOME, "config.json"), encoding="utf-8") as f:
            cfg.update({k: v for k, v in json.load(f).items() if k in DEFAULTS})
    except FileNotFoundError:
        pass
    return cfg


class State:
    """Timestamps kept between runs in state.json: when a repeating line was last logged (so it is logged once per
    QUIET_MINUTES), when ComfyUI was last seen busy, when the idle check last read the VM."""

    def __init__(self):
        self.path = os.path.join(HOME, "state.json")
        try:
            with open(self.path, encoding="utf-8") as f:
                self.last = json.load(f)
        except (FileNotFoundError, ValueError):
            self.last = {}

    def due(self, key, minutes=QUIET_MINUTES):
        if time.time() - self.last.get(key, 0) < minutes * 60:
            return False
        self.set(key, time.time())
        return True

    def set(self, key, value):
        self.last[key] = value
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.last, f)


def make_log():
    os.makedirs(HOME, exist_ok=True)
    log = logging.getLogger("memory-watchdog")
    handler = logging.handlers.RotatingFileHandler(os.path.join(HOME, "watchdog.log"), maxBytes=256 * 1024,
                                                   backupCount=2, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    return log


def status(cfg):
    free = windows_available_gb()
    running = vm_running(cfg["distro"])
    cache = vm_cache_gb(cfg["distro"]) if running else 0.0
    print(f"Windows available: {free:.1f} GB (acts below {cfg['free_below_gb']} GB)")
    print(f"{cfg['distro']}: {'running' if running else 'not running'}, file cache {cache:.1f} GB "
          f"(acts from {cfg['min_cache_gb']} GB)")
    if cfg["idle_minutes"]:
        busy = comfy_busy(cfg)
        since = State().last.get("comfy-busy")
        idle = "busy now" if busy else f"idle {(time.time() - since) / 60:.0f} min" if since else "idle"
        print(f"ComfyUI: {idle} (drops from {cfg['idle_cache_gb']} GB of cache after {cfg['idle_minutes']} idle min)")
    else:
        print("Idle drop: off (idle_minutes is 0)")
    print("Would drop the cache now." if running and should_drop(free, cache, cfg) else "Nothing to do now.")
    print("\n".join(vram_status(vram_report(cfg), cfg)))


def run(cfg, log):
    free = windows_available_gb()
    if free >= cfg["free_below_gb"]:
        return                                   # the usual case: no wsl call, no log line
    quiet = State()
    distro = cfg["distro"]
    if not vm_running(distro):
        if quiet.due("not-running"):
            log.info(f"low memory ({free:.1f} GB free) but {distro} is not running; nothing to drop")
        return
    cache = vm_cache_gb(distro)
    if not should_drop(free, cache, cfg):
        if quiet.due("small-cache"):
            log.info(f"low memory ({free:.1f} GB free) but {distro} holds only {cache:.1f} GB of cache; left alone")
        return
    drop(distro, free, cache, "low memory", log)


def idle_reclaim(cfg, log):
    """Drops the cache a finished job left behind once ComfyUI has been idle for idle_minutes."""
    if not cfg["idle_minutes"]:
        return
    state, now = State(), time.time()
    if comfy_busy(cfg) or "comfy-busy" not in state.last:
        state.set("comfy-busy", now)
        return
    idle = (now - state.last["comfy-busy"]) / 60
    if idle < cfg["idle_minutes"] or not state.due("idle-check", IDLE_CHECK_MINUTES):
        return
    distro = cfg["distro"]
    if not vm_running(distro):
        return
    cache = vm_cache_gb(distro)
    if idle_drop_due(idle, cache, cfg):
        drop(distro, windows_available_gb(), cache, f"ComfyUI idle {idle:.0f} min", log)


def drop(distro, free, cache, why, log):
    started = time.monotonic()
    wsl("-d", distro, "-u", "root", "sh", "-c", "sync; echo 1 > /proc/sys/vm/drop_caches", timeout=120)
    took = time.monotonic() - started
    time.sleep(15)                               # WSL hands freed pages back to Windows over ~10 s
    log.info(f"dropped {distro} cache ({why}) in {took:.1f} s: Windows free {free:.1f} -> "
             f"{windows_available_gb():.1f} GB, VM cache {cache:.1f} -> {vm_cache_gb(distro):.1f} GB")


def main():
    cfg = load_config()
    if "--status" in sys.argv:
        status(cfg)
        return
    log = make_log()
    deadline = time.monotonic() + RUN_SECONDS   # a model load fills ~14 GB in under a minute: look every 10 s
    while True:
        for check in (run, idle_reclaim, check_vram):
            try:
                check(cfg, log)
            except Exception as e:  # never crash silently or loudly: log it, at most once per QUIET_MINUTES
                if State().due(f"error:{check.__name__}:{type(e).__name__}"):
                    log.info(f"error in {check.__name__}: {type(e).__name__}: {e}")
        if time.monotonic() + CHECK_SECONDS > deadline:
            break
        time.sleep(CHECK_SECONDS)


if __name__ == "__main__":
    main()
