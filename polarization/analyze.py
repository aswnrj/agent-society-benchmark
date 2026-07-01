"""Classify the gun-control opinion change of each citizen into the four
categories reported in Figure 16 of the AgentSociety paper (Section 7.2):
More Polarized, Less Polarized, Opposed, and Unchanged.

Each arm saves the initial and final attitudes (0 = oppose, 10 = support) via
the SAVE_CONTEXT workflow step into
``agentsociety_data/exps/<tenant>/<exp>/artifacts.json``.
"""

import argparse
import glob
import json
import os
import sys

TOPIC = "Whether to support stronger gun control?"
NEUTRAL = 5.0
CATEGORIES = ["More Polarized", "Less Polarized", "Opposed", "Unchanged"]


def attitude(value):
    return float(value[TOPIC])


def classify(initial, final):
    if (initial < NEUTRAL < final) or (final < NEUTRAL < initial):
        return "Opposed"
    before, after = abs(initial - NEUTRAL), abs(final - NEUTRAL)
    if after > before:
        return "More Polarized"
    if after < before:
        return "Less Polarized"
    return "Unchanged"


def find_run(path):
    if os.path.isfile(path):
        return path
    runs = []
    for f in glob.glob(os.path.join(path, "**", "artifacts.json"), recursive=True):
        try:
            data = json.load(open(f))
        except (OSError, json.JSONDecodeError):
            continue
        if "guncontrol_attitude_initial" in data and "guncontrol_attitude_final" in data:
            runs.append((os.path.getmtime(f), f))
    return max(runs)[1] if runs else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="agentsociety_data")
    parser.add_argument("--csv")
    args = parser.parse_args()

    run = find_run(args.path)
    if run is None:
        sys.exit("No completed run found (need both initial and final attitudes).")

    data = json.load(open(run))
    initial = data["guncontrol_attitude_initial"]
    final = data["guncontrol_attitude_final"]

    counts = dict.fromkeys(CATEGORIES, 0)
    rows = []
    for agent_id in initial:
        if agent_id not in final:
            continue
        a0, a1 = attitude(initial[agent_id]), attitude(final[agent_id])
        category = classify(a0, a1)
        counts[category] += 1
        rows.append((int(agent_id), a0, a1, category))

    total = len(rows)
    print(f"{run}\n{total} citizens\n")
    for category in CATEGORIES:
        print(f"{category:<15} {counts[category]:3d}  {100 * counts[category] / total:5.1f}%")

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("agent_id,initial,final,category\n")
            for agent_id, a0, a1, category in sorted(rows):
                f.write(f"{agent_id},{a0},{a1},{category}\n")


if __name__ == "__main__":
    main()
