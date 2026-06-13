from pathlib import Path

from codegen.parser import parse


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_kernel(filename: str):
    program = parse(PROJECT_ROOT / "testdata" / filename)
    assert program is not None
    return program
