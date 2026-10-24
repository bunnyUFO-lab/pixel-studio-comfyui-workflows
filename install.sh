#!/bin/sh
# Installer for Linux (and Git Bash): finds or clones the repository, runs `pixel-studio-comfy setup`, and puts the
# command on PATH. All checks and file writes are in the Python CLI (psc/); this script only finds things.
#
#   sh install.sh [--dir PATH] [--ref BRANCH] [--no-path] [setup options, e.g. --preset svg --yes]
set -eu

REPO_URL="https://github.com/bunnyUFO-lab/pixel-studio-comfyui-workflows"
dir=""
ref="main"
add_path=1

# Consume --dir, --ref and --no-path; everything else is forwarded to setup (kept in "$@").
count=$#
while [ "$count" -gt 0 ]; do
    arg=$1
    shift
    count=$((count - 1))
    case "$arg" in
        --dir) dir=${1:?--dir needs a path}; shift; count=$((count - 1)) ;;
        --dir=*) dir=${arg#--dir=} ;;
        --ref) ref=${1:?--ref needs a branch}; shift; count=$((count - 1)) ;;
        --ref=*) ref=${arg#--ref=} ;;
        --no-path) add_path=0 ;;
        *) set -- "$@" "$arg" ;;
    esac
done

is_checkout() {
    [ -f "$1/docker-compose.yml" ] && [ -d "$1/psc" ]
}

if [ -z "$dir" ]; then
    here=$(cd "$(dirname "$0")" && pwd)
    if is_checkout "$here"; then
        dir=$here
    else
        dir="$HOME/pixel-studio-comfy"
    fi
fi

if is_checkout "$dir"; then
    echo "Using the checkout in $dir"
elif [ -e "$dir" ]; then
    echo "error: $dir exists but is not a pixel-studio-comfyui-workflows checkout. Choose another --dir." >&2
    exit 1
else
    if ! command -v git >/dev/null 2>&1; then
        echo "error: git is needed to download the repository. Install git, then run this again." >&2
        exit 1
    fi
    echo "Cloning $REPO_URL ($ref) into $dir"
    git clone --branch "$ref" "$REPO_URL" "$dir"
fi

py=""
for cand in python3 python "py -3"; do
    # Unquoted on purpose: "py -3" is two words. The Windows Store python3 alias exists but fails this test.
    # shellcheck disable=SC2086
    if $cand -c "import sys; sys.exit(sys.version_info < (3, 9))" >/dev/null 2>&1; then
        py=$cand
        break
    fi
done
if [ -z "$py" ]; then
    echo "error: Python 3.9 or newer was not found." >&2
    echo "  Debian/Ubuntu: sudo apt install python3     Fedora: sudo dnf install python3     macOS: brew install python" >&2
    exit 1
fi

rc=0
# shellcheck disable=SC2086
PYTHONPATH="$dir${PYTHONPATH:+:$PYTHONPATH}" $py -m psc --dir "$dir" setup "$@" || rc=$?

if [ "$add_path" -eq 1 ]; then
    bin="$HOME/.local/bin"
    shim="$bin/pixel-studio-comfy"
    want="exec \"$dir/bin/pixel-studio-comfy\" \"\$@\""
    case "$dir" in
        *'"'* | *'$'* | *'`'* | *'\'*)
            echo "NOTE: not writing $shim: the folder name has characters a shell shim cannot quote." ;;
        *)
            if [ -e "$shim" ] && [ "$(sed -n 2p "$shim" 2>/dev/null)" != "$want" ]; then
                echo "NOTE: $shim already points somewhere else; left unchanged. Run $dir/bin/pixel-studio-comfy instead."
            else
                mkdir -p "$bin"
                printf '#!/bin/sh\n%s\n' "$want" > "$shim"
                chmod 755 "$shim"
                echo "Installed the command: $shim"
            fi
            case ":$PATH:" in
                *":$bin:"*) ;;
                *) echo "NOTE: $bin is not on your PATH. Add it (for example in ~/.profile): export PATH=\"$bin:\$PATH\"" ;;
            esac ;;
    esac
fi

exit "$rc"
