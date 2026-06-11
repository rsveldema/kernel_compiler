"""Compiler frontend"""

import sys as _sys_mod
import os

# Ensure project root is on path so 'import codegen' works, and remove the
# script's own directory so stdlib imports (e.g. ast, logging) don't resolve to
# our local packages first.  Python adds the script directory to sys.path[0] at
# startup; we must neutralise it before any standard-library imports run.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in _sys_mod.path:
    _sys_mod.path.insert(0, _project_root)

_stdlib_paths = []
for _sp in _sys_mod.path:
    norm = os.path.normpath(_sp)
    if norm == os.path.normpath(os.path.dirname(os.path.abspath(__file__))):
        continue
    _stdlib_paths.append(_sp)
if len(_stdlib_paths) < len(_sys_mod.path):
    _sys_mod.path[:] = _stdlib_paths

import argparse
import subprocess
import logging

from codegen.parser import parser
from codegen.ast.program import Program
from codegen.transforms import transform
from codegen.visitors.pretty_printer import prettyprint
from codegen.visitors.vulkan_kernel_visitor import VulkanKernelVisitor
from codegen.visitors.vulkan_cpp_stub_visitor import VulkanCppStubVisitor
from codegen.optim import perform_blocking

log = logging.getLogger(__name__)


def read_file(filename: str) -> str:
    with open(filename) as f:
        return f.read()


def optimize(program: Program, enable_optimizations: bool = True, chunk_size: int = 8) -> Program:
    if enable_optimizations:
        program = perform_blocking(program, chunk_size)
    from codegen.visitors.resolve_array_indices import resolve_array_indices
    program = resolve_array_indices(program)
    return program


def compile(filename: str, enable_optimizations: bool = True, chunk_size: int = 8) -> Program:
    print(f"--------------- parsing: {filename} -----------------")
    text = read_file(filename)
    ret = parser.parse(text)
    program = transform(ret)
    program._source_filename = filename
    program = optimize(program, enable_optimizations, chunk_size)
    return program


def generate_vulkan(filename: str, output: str, enable_optimizations: bool = True, chunk_size: int = 8) -> None:
    program = compile(filename, enable_optimizations, chunk_size)
    visitor = VulkanKernelVisitor()
    shader = program.accept(visitor)
    with open(output, "w") as f:
        f.write(shader)
    print(f"Generated Vulkan shader -> {output}")


def compile_vulkan(input_file: str, output_spv: str, enable_optimizations: bool = True, chunk_size: int = 8) -> None:
    glsl_path = input_file.rsplit(".", 1)[0] + ".glsl"
    generate_vulkan(input_file, glsl_path, enable_optimizations, chunk_size)
    cmd = [
        "glslc", "-fshader-stage=compute", "-o", output_spv,
        "--target-env=vulkan1.2", glsl_path,
    ]
    print(f"running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Compilation failed:")
        print(result.stderr, file=_sys_mod.stderr)
        _sys_mod.exit(1)
    print(f"Compiled SPIR-V -> {output_spv}")


def generate_cpp_stub(filename: str, output: str, enable_optimizations: bool = True, chunk_size: int = 8) -> None:
    program = compile(filename, enable_optimizations, chunk_size)
    visitor = VulkanCppStubVisitor()
    stub = program.accept(visitor)
    with open(output, "w") as f:
        f.write(stub)
    print(f"Generated C++ stub -> {output}")


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(description="Kernel compiler front-end")
    _parser.add_argument(
        "file", nargs="?", help="Input .kernel file (required for --vulkan/--compile)"
    )
    _parser.add_argument(
        "--vulkan", metavar="OUTPUT", help="Generate Vulkan GLSL shader to OUTPUT"
    )
    _parser.add_argument(
        "--compile",
        metavar="OUTPUT_SPV",
        help="Generate and compile Vulkan shader to SPIR-V",
    )
    _parser.add_argument(
        "--cpp-stub",
        metavar="OUTPUT_HPP",
        help="Generate C++ stub header for kernel dispatch",
    )
    _parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Disable optimization passes before code generation",
    )
    _parser.add_argument(
        "--chunk-size",
        type=int,
        default=8,
        help="Reduction chunk size for shared-memory tiling optimizations",
    )
    args = _parser.parse_args()
    enable_optimizations = not args.no_optimize
    if args.chunk_size <= 0:
        _parser.error("--chunk-size must be positive")

    if args.vulkan or args.compile:
        if not args.file:
            _parser.error("--vulkan and --compile require an input FILE argument")
        if args.vulkan:
            generate_vulkan(args.file, args.vulkan, enable_optimizations, args.chunk_size)
        elif args.compile:
            compile_vulkan(args.file, args.compile, enable_optimizations, args.chunk_size)
    elif args.cpp_stub:
        if not args.file:
            _parser.error("--cpp-stub requires an input FILE argument")
        generate_cpp_stub(args.file, args.cpp_stub, enable_optimizations, args.chunk_size)
    else:
        # Default: prettyprint all files
        for path in _sys_mod.argv[1:]:
            program = compile(path, enable_optimizations, args.chunk_size)
            print(prettyprint(program))
