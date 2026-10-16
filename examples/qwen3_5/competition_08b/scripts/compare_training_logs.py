import argparse
import json
import re
from pathlib import Path

ROW = re.compile(
    r"iteration\s+(?P<step>\d+)/\s*\d+.*?"
    r"elapsed time per iteration \(ms\):\s*(?P<ms>[0-9.]+).*?"
    r"global batch size:\s*(?P<gbs>\d+).*?loss:\s*(?P<loss>[0-9.Ee+-]+)"
)


def parse(path):
    rows = {}
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    for match in ROW.finditer(text):
        value = match.groupdict()
        rows[int(value["step"])] = {
            "elapsed_ms": float(value["ms"]),
            "global_batch_size": int(value["gbs"]),
            "loss": float(value["loss"]),
        }
    return rows


def summarize(rows, start, end):
    selected = [rows[i] for i in range(start, end + 1) if i in rows]
    if not selected:
        raise ValueError(f"no iterations in [{start}, {end}]")
    mean_ms = sum(x["elapsed_ms"] for x in selected) / len(selected)
    gbs = selected[0]["global_batch_size"]
    return {
        "iterations": len(selected),
        "mean_elapsed_ms": mean_ms,
        "samples_per_second": gbs * 1000.0 / mean_ms,
        "mean_loss": sum(x["loss"] for x in selected) / len(selected),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ascendc", required=True)
    parser.add_argument("--triton", required=True)
    parser.add_argument("--start", type=int, default=11)
    parser.add_argument("--end", type=int, default=100)
    parser.add_argument("--output")
    args = parser.parse_args()
    ascendc, triton = parse(args.ascendc), parse(args.triton)
    common = [i for i in range(args.start, args.end + 1) if i in ascendc and i in triton]
    if not common:
        raise SystemExit("no common iterations")
    diffs = [abs(ascendc[i]["loss"] - triton[i]["loss"]) for i in common]
    a = summarize(ascendc, args.start, args.end)
    t = summarize(triton, args.start, args.end)
    result = {
        "window": {"start": args.start, "end": args.end, "common_iterations": len(common)},
        "ascendc": a,
        "triton": t,
        "loss_alignment": {
            "mean_absolute_difference": sum(diffs) / len(diffs),
            "max_absolute_difference": max(diffs),
            "last_step_absolute_difference": diffs[-1],
        },
        "ascendc_vs_triton": {
            "throughput_change_percent": (a["samples_per_second"] / t["samples_per_second"] - 1) * 100,
            "step_time_change_percent": (a["mean_elapsed_ms"] / t["mean_elapsed_ms"] - 1) * 100,
        },
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
