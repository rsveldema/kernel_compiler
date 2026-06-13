#!/usr/bin/env python3
"""Generate pytest-compatible test files for all offload_parfor_*.kernel files in testdata/."""

import pathlib

TEST_TEMPLATE = '''from unittest_offload_parfor_helper import parse_kernel


def test_parse():
    parse_kernel("{kernel_name}")
'''


def main():
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    testdata_dir = repo_root / "testdata"
    tests_dir = repo_root / "tests"

    # Remove stale test files first
    for f in tests_dir.glob("test_offload_parfor_*_cc.py"):
        f.unlink()

    kernels = sorted(testdata_dir.glob("offload_parfor_*.kernel"))
    print(f"Generating {len(kernels)} test files...")

    for kernel_file in kernels:
        base = kernel_file.name.rsplit(".kernel", 1)[0]
        stem = base.rsplit(".cc", 1)[0]
        test_name = f"test_{stem}_cc.py"

        (tests_dir / test_name).write_text(
            TEST_TEMPLATE.format(kernel_name=kernel_file.name)
        )
        print(f"  {test_name}")


if __name__ == "__main__":
    main()
