#!/usr/bin/env python3
"""nanoSim Experiment-Labor — Einstiegspunkt (dünner Shim).

Die gesamte Logik liegt im getesteten Paket `nanosim.lab`. Dieses Skript ruft
nur den gehärteten Runner auf.

Start (aus dem Repo, mit aktivem venv):
    LAB_EXPERIMENT=all nohup .venv/bin/python experiments/lab.py \
        > experiments/results/nohup.out 2>&1 &

Status aus der Ferne abfragen:
    .venv/bin/python experiments/lab.py status
"""

import sys

from nanosim.lab.runner import main, read_status

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        print(read_status())
    else:
        main()
