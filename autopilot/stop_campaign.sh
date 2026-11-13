#!/usr/bin/env bash
# Stop a running campaign: first the shells that launched it (so nothing chained after it
# starts), then the Python process itself. This script's own command line doesn't contain
# the pattern, so it can't kill itself.
pat='autopilot/campaign[.]py'
for pid in $(ps -eo pid,args | grep -E "$pat" | grep -E "bin/bash" | awk '{print $1}'); do kill "$pid" 2>/dev/null; done
for pid in $(ps -eo pid,args | grep -E "$pat" | grep -E "python" | grep -v "bin/bash" | awk '{print $1}'); do kill "$pid" 2>/dev/null; done
sleep 1
ps -eo pid,args | grep -E "$pat" | grep -v grep || echo "campaign stopped"
