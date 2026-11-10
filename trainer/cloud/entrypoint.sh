#!/usr/bin/env bash
# Runs the trainer runner and, whatever its exit code, asks the host to end the machine (shutdown.py).
# A pod started without RUNNER_TOKEN makes the runner exit 2, and the machine ends here too.
set -u

# The coreutils timeout is the backstop if the runner itself hangs.
timeout --signal=TERM --kill-after=60 $(( ${RUNNER_TIME_LIMIT_MAX_S:-86400} + 600 )) python -u /runner/runner.py &
child=$!
trap 'kill -TERM "$child" 2>/dev/null' TERM
trap 'kill -INT "$child" 2>/dev/null' INT

wait "$child"
code=$?
# a trapped signal ends the first wait early: wait again for the real exit code
while kill -0 "$child" 2>/dev/null; do
  wait "$child"
  code=$?
done
echo "runner: exited with code $code"

python -u /runner/shutdown.py
exit "$code"
