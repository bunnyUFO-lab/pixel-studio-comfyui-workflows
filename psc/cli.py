"""Root of the pixel-studio-comfy command line: global options, the command registry and dispatch.

Extending it (for example the M3 cloud commands) needs no change to the commands that exist:

* add a module with `register(subparsers)`; it adds parsers and calls `set_defaults(func=handler)` on each
* a handler takes `(args, stack)` and returns an exit code
* append the module to COMMAND_MODULES below
* a command group (`cloud modal deploy`) is a parser with its own subparsers: `core.group(subparsers, "cloud")`
  returns the group, creating it the first time, so any number of modules can add to the same one.
"""
import argparse
import os
import sys
from pathlib import Path

from psc import __version__, local, models_cmd, setup, update_cmd
from psc.stack import CliError, Stack

COMMAND_MODULES = [local, models_cmd, setup, update_cmd]

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_parser(modules=None):
    parser = argparse.ArgumentParser(
        prog="pixel-studio-comfy",
        description="Install, check and run the ComfyUI stack of this repository.")
    parser.add_argument("--version", action="version", version="pixel-studio-comfy " + __version__)
    parser.add_argument("--dir", metavar="PATH",
                        help="stack folder with docker-compose.yml (env PSC_DIR; default: the repo this CLI lives in)")
    parser.add_argument("--url", metavar="URL",
                        help="ComfyUI base URL (env PSC_URL; default http://127.0.0.1:<PORT from .env, else 8188>)")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True
    for module in COMMAND_MODULES if modules is None else modules:
        module.register(sub)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    root = args.dir or os.environ.get("PSC_DIR") or str(REPO_ROOT)
    stack = Stack(root, args.url or os.environ.get("PSC_URL") or None)
    try:
        return args.func(args, stack) or 0
    except KeyboardInterrupt:
        return 130
    except CliError as e:
        print("error: %s" % e, file=sys.stderr)
        return 1
