import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--gdn", choices=("ascendc", "triton"), required=True)
    args = parser.parse_args()
    text = Path(args.source).read_text(encoding="utf-8")
    old = "  gdn_implementation: ascendc"
    if old not in text:
        raise SystemExit(f"expected exactly one baseline setting: {old}")
    text = text.replace(old, f"  gdn_implementation: {args.gdn}", 1)
    Path(args.destination).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
