"""Compiler frontend"""

import sys as _sys_mod
import os

# Ensure project root is on path so 'import codegen' works, and remove the
# script's own directory so stdlib imports don't resolve to local packages first.
# Python adds the script directory to sys.path[0] at
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
from codegen.kast.program import Program
from codegen.transforms import transform
from codegen.visitors.pretty_printer import prettyprint
from codegen.visitors.vulkan_kernel_visitor import VulkanKernelVisitor
from codegen.visitors.vulkan_cpp_stub_visitor import VulkanCppStubVisitor
from codegen.visitors.rllm_vulkan_dispatch_stub_visitor import RllmVulkanDispatchStubVisitor
from codegen.optim import perform_blocking, perform_cooperative_matrix2

log = logging.getLogger(__name__)


def read_file(filename: str) -> str:
    with open(filename) as f:
        return f.read()


def optimize(
    program: Program,
    enable_optimizations: bool = True,
    chunk_size: int = 8,
    optimization_pass: str = "shared-memory",
) -> Program:
    if enable_optimizations:
        if optimization_pass == "coopmat2":
            program = perform_cooperative_matrix2(program, chunk_size)
        else:
            program = perform_blocking(program, chunk_size)
    from codegen.visitors.resolve_array_indices import resolve_array_indices
    program = resolve_array_indices(program)
    return program


def compile(
    filename: str,
    enable_optimizations: bool = True,
    chunk_size: int = 8,
    optimization_pass: str = "shared-memory",
) -> Program:
    print(f"--------------- parsing: {filename} -----------------")
    text = read_file(filename)
    ret = parser.parse(text)
    program = transform(ret)
    program._source_filename = filename
    program = optimize(program, enable_optimizations, chunk_size, optimization_pass)
    return program


def generate_vulkan(
    filename: str,
    output: str,
    enable_optimizations: bool = True,
    chunk_size: int = 8,
    optimization_pass: str = "shared-memory",
    rllm_dispatch_stub: str | None = None,
    rllm_spv_path: str | None = None,
    use_bfloat16: bool = False,
) -> None:
    program = compile(filename, enable_optimizations, chunk_size, optimization_pass)
    visitor = VulkanKernelVisitor(use_bfloat16=use_bfloat16)
    shader = program.accept(visitor)
    with open(output, "w") as f:
        f.write(shader)
    print(f"Generated Vulkan shader -> {output}")
    visitor = VulkanCppStubVisitor(use_bfloat16=use_bfloat16)
    stub = program.accept(visitor)
    stub_output = output.rsplit(".", 1)[0] + ".h"
    with open(stub_output, "w") as f:
        f.write(stub)
    print(f"Generated C++ stub -> {stub_output}")
    if rllm_dispatch_stub:
        visitor = RllmVulkanDispatchStubVisitor(rllm_spv_path or (output.rsplit(".", 1)[0] + ".spv"))
        dispatch_stub = program.accept(visitor)
        with open(rllm_dispatch_stub, "w") as f:
            f.write(dispatch_stub)
        print(f"Generated RLLM dispatch stub -> {rllm_dispatch_stub}")


def compile_vulkan(
    input_file: str,
    output_spv: str,
    enable_optimizations: bool = True,
    chunk_size: int = 8,
    optimization_pass: str = "shared-memory",
    rllm_dispatch_stub: str | None = None,
    rllm_spv_path: str | None = None,
    use_bfloat16: bool = False,
) -> None:
    glsl_path = input_file.rsplit(".", 1)[0] + ".glsl"
    generate_vulkan(input_file, glsl_path, enable_optimizations, chunk_size, optimization_pass, rllm_dispatch_stub, rllm_spv_path, use_bfloat16=use_bfloat16)
    cmd = [
        "glslc", "-fshader-stage=compute", "-o", output_spv,
        "--target-env=vulkan1.4" if optimization_pass == "coopmat2" else "--target-env=vulkan1.2", glsl_path,
    ]
    print(f"running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Compilation failed:")
        print(result.stderr, file=_sys_mod.stderr)
        _sys_mod.exit(1)
    print(f"Compiled SPIR-V -> {output_spv}")


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
    _parser.add_argument("--rllm-dispatch-stub", metavar="OUTPUT_H", help="Generate an RLLM Vulkan dispatch wrapper header")
    _parser.add_argument("--rllm-spv-path", metavar="REL_SPV", help="SPIR-V path embedded in the RLLM dispatch wrapper")
    _parser.add_argument(
        "--bfloat16",
        action="store_true",
        help="Generate bfloat (16-bit) instead of float16 in Vulkan/C++ output",
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
    _parser.add_argument(
        "--coopmat2",
        action="store_true",
        help="Use the VK_NV_cooperative_matrix2 optimization pass",
    )
    args = _parser.parse_args()
    enable_optimizations = not args.no_optimize
    use_bfloat16 = getattr(args, "bfloat16", False)
    optimization_pass = "coopmat2" if args.coopmat2 else "shared-memory"
    if args.chunk_size <= 0:
        _parser.error("--chunk-size must be positive")

    if args.vulkan or args.compile:
        if not args.file:
            _parser.error("--vulkan and --compile require an input FILE argument")
        if args.vulkan:
            generate_vulkan(args.file, args.vulkan, enable_optimizations, args.chunk_size, optimization_pass, args.rllm_dispatch_stub, args.rllm_spv_path, use_bfloat16=use_bfloat16)
        elif args.compile:
            compile_vulkan(args.file, args.compile, enable_optimizations, args.chunk_size, optimization_pass, args.rllm_dispatch_stub, args.rllm_spv_path, use_bfloat16=use_bfloat16)
    else:
        # Default: prettyprint all files
        for path in _sys_mod.argv[1:]:
            program = compile(path, enable_optimizations, args.chunk_size, optimization_pass)
            print(prettyprint(program))
