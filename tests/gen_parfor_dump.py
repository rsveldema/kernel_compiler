#!/usr/bin/env python3
"""Generate a PARFOR dump file from a source .kernel file.

Reads a .kernel file containing an OFFLOAD_PARFOR block and writes a
dump copy that adds const-int loop-variable initialisations after
BEGIN, so that compile.py sees defined loop variables.

Usage:
    python gen_parfor_dump.py <input.kernel> <output.kernel>
"""

import argparse
import re
import sys


_DIM_NAMES = ("x", "y", "z")


def parse_kernel(path: str) -> dict:
    """Parse a .kernel file and return structured components."""
    with open(path) as f:
        lines = f.readlines()

    result = {
        "program_line": None,
        "parfor_line": None,
        "param_lines": [],
        "begin_line": None,
        "body_lines": [],
        "end_line": None,
        "lineno": 0,
    }

    state = "initial"  # initial, params, begin, body, end

    for raw_line in lines:
        line = raw_line.rstrip("\n")

        # PROGRAM("...")
        if state == "initial":
            m = re.match(r'^PROGRAM\("(.*?):(\d+)"\)\s*$', line)
            if m:
                result["program_line"] = line
                result["lineno"] = int(m.group(2))
                continue
            m = re.match(r'^PARAMETERS\s*$', line, re.IGNORECASE)
            if m:
                state = "params"
                continue

        # PARFOR invocation (between PROGRAM and PARAMETERS, or after PROGRAM)
        if state == "initial" and line.strip():
            m = re.match(r'^OFFLOAD_PARFOR_(?:3D_PARAM|2D_PARAM|1D_PARAM)\(', line)
            if m:
                result["parfor_line"] = line
                continue

        # PARAMETERS block
        if state == "params":
            if line.strip() == "BEGIN":
                state = "begin"
                result["begin_line"] = line
                continue
            result["param_lines"].append(line)

        # BEGIN / body
        if state == "begin":
            state = "body"

        if state == "body":
            if line.strip() == "END_PROGRAM":
                result["end_line"] = line
                result["body_lines"].append(line)
                state = "end"
                continue
            result["body_lines"].append(line)

        # Skip other lines in initial state (blank lines, etc.)
        if state == "initial" and not line.strip():
            continue

    return result


def extract_parfor_vars(parfor_line: str) -> list[str]:
    """Extract loop variable names from an OFFLOAD_PARFOR_*_PARAM line."""
    m = re.search(r'OFFLOAD_PARFOR_\w+\(([^)]+)\)', parfor_line)
    if not m:
        return []
    first_part = m.group(1).split(",")[0].strip()
    return [v.strip() for v in first_part.split(",") if v.strip()]


def generate_dump(parsed: dict, output_path: str) -> None:
    """Write the dump file with PARFOR init lines after BEGIN."""
    lines = []
    lines.append(parsed["program_line"])
    lines.append("")

    # Original PARFOR invocation
    if parsed["parfor_line"]:
        lines.append(parsed["parfor_line"])
        lines.append("")

    # PARAMETERS block
    lines.append("PARAMETERS")
    for pl in parsed["param_lines"]:
        lines.append(pl)
    lines.append("")

    # BEGIN
    lines.append("BEGIN")

    # Add loop variable initializations
    vars_ = extract_parfor_vars(parsed["parfor_line"] or "")
    for idx, var_name in enumerate(vars_):
        dim = _DIM_NAMES[min(idx, 2)]
        lines.append(f"const int {var_name} = int(gl_GlobalInvocationID.{dim});")

    # Body
    for bl in parsed["body_lines"]:
        if bl.strip() == "END_PROGRAM":
            break
        lines.append(bl)
    if not any(bl.strip() == "END_PROGRAM" for bl in parsed["body_lines"]):
        lines.append("")
    lines.append(parsed["end_line"] or "END_PROGRAM")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a PARFOR dump file from a source .kernel file."
    )
    parser.add_argument("input", help="Input .kernel file")
    parser.add_argument("output", help="Output dump .kernel file")
    args = parser.parse_args()

    parsed = parse_kernel(args.input)
    output_dir = __import__("pathlib").Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_dump(parsed, args.output)


if __name__ == "__main__":
    main()
