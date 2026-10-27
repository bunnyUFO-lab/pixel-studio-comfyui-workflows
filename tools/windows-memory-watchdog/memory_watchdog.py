"""Hands the Docker VM's file cache back to Windows when Windows runs short of RAM.

ComfyUI in Docker Desktop runs inside a WSL2 VM. Loading a big model (Wan 2.x stages ~14 GB) fills
the VM with file cache, Windows drops below 1 GB free and the job sits at "Model Initializing" for
minutes. WSL's own autoMemoryReclaim only reclaims once the VM is idle. This script, started every
minute by the "ComfyUI memory watchdog" scheduled task (install.ps1), looks every 10 seconds and
drops that cache as soon as Windows' available RAM falls below a threshold and the VM holds enough
cache to matter:

    wsl -d docker-desktop -u root sh -c "sync; echo 1 > /proc/sys/vm/drop_caches"

Only clean cache goes; files ComfyUI is using stay mapped. It never stops ComfyUI, Docker or WSL,
and it does not start the VM when it is off. The task runs it with pythonw.exe and its children
with CREATE_NO_WINDOW, so no window ever opens or takes focus.

Settings: config.json beside the installed copy (keys as in DEFAULTS) overrides the defaults.
Log: %LOCALAPPDATA%\\memory-watchdog\\watchdog.log, one line per action, rotated at 256 KB.

    python memory_watchdog.py --status   # print the numbers and what it would do; change nothing
"""
import ctypes
import json
import logging
import logging.handlers
import os
import subprocess
import sys
import time

DEFAULTS = {
    "free_below_gb": 4.0,   # act when Windows' available RAM is below this
    "min_cache_gb": 2.0,    # ...and the VM holds at least this much droppable file cache
    "distro": "docker-desktop",
}
HOME = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "memory-watchdog")
CHECK_SECONDS = 10          # the task starts this every minute; it checks this often until the next start
RUN_SECONDS = 50
QUIET_MINUTES = 30          # repeat a "low but nothing to drop" or error line at most this often
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


def load_config():
    cfg = dict(DEFAULTS)
    try:
        with open(os.path.join(HOME, "config.json"), encoding="utf-8") as f:
            cfg.update({k: v for k, v in json.load(f).items() if k in DEFAULTS})
    except FileNotFoundError:
        pass
    return cfg


class Quiet:
    """Remembers between runs when a repeating line was last logged, so it is logged once per QUIET_MINUTES."""

    def __init__(self):
        self.path = os.path.join(HOME, "state.json")
        try:
            with open(self.path, encoding="utf-8") as f:
                self.last = json.load(f)
        except (FileNotFoundError, ValueError):
            self.last = {}

    def due(self, key):
        if time.time() - self.last.get(key, 0) < QUIET_MINUTES * 60:
            return False
        self.last[key] = time.time()
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.last, f)
        return True


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
    print("Would drop the cache now." if running and should_drop(free, cache, cfg) else "Nothing to do now.")


def run(cfg, log):
    free = windows_available_gb()
    if free >= cfg["free_below_gb"]:
        return                                   # the usual case: no wsl call, no log line
    quiet = Quiet()
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
    started = time.monotonic()
    wsl("-d", distro, "-u", "root", "sh", "-c", "sync; echo 1 > /proc/sys/vm/drop_caches", timeout=120)
    took = time.monotonic() - started
    time.sleep(15)                               # WSL hands freed pages back to Windows over ~10 s
    log.info(f"dropped {distro} cache in {took:.1f} s: Windows free {free:.1f} -> {windows_available_gb():.1f} GB, "
             f"VM cache {cache:.1f} -> {vm_cache_gb(distro):.1f} GB")


def main():
    cfg = load_config()
    if "--status" in sys.argv:
        status(cfg)
        return
    log = make_log()
    deadline = time.monotonic() + RUN_SECONDS   # a model load fills ~14 GB in under a minute: look every 10 s
    while True:
        try:
            run(cfg, log)
        except Exception as e:  # never crash silently or loudly: log it, at most once per QUIET_MINUTES
            if Quiet().due("error:" + type(e).__name__):
                log.info(f"error: {type(e).__name__}: {e}")
        if time.monotonic() + CHECK_SECONDS > deadline:
            break
        time.sleep(CHECK_SECONDS)


if __name__ == "__main__":
    main()
