#!/usr/bin/env bash
# Runs the trainer runner and, whatever its exit code, asks the host to end the machine (shutdown.py).
# A pod started without RUNNER_TOKEN makes the runner exit 2, and the machine ends here too.
# shutdown.py never exits when it cannot end the machine (an exited container is restarted by RunPod and would loop).
set -u

# This image is for a public address: a stray RUNNER_ALLOW_OPEN in the pod's environment must not open the runner.
unset RUNNER_ALLOW_OPEN

# The coreutils timeout is the backstop if the runner itself hangs.
timeout --signal=TERM --kill-after=60 $(( ${RUNNER_TIME_LIMIT_MAX_S:-86400} + 600 )) python -u /runner/runner.py &
child=$!
signalled=0
trap 'signalled=1; kill -TERM "$child" 2>/dev/null' TERM
trap 'signalled=1; kill -INT "$child" 2>/dev/null' INT

wait "$child"
code=$?
# a trapped signal ends the first wait early: wait again for the real exit code
while kill -0 "$child" 2>/dev/null; do
  wait "$child"
  code=$?
done
echo "runner: exited with code $code"

# After a stop signal Docker allows only about 10 s: keep each try short.
if [ "$signalled" = 1 ]; then export RUNNER_SHUTDOWN_TIMEOUT_S=5; fi
python -u /runner/shutdown.py
exit "$code"
