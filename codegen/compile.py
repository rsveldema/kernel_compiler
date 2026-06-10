"""Compiler frontend"""

import sys as _sys_mod
import argparse
import subprocess
import logging

from codegen.parser import parser
from codegen.ast.program import Program
from codegen.transforms import transform
from codegen.visitors.pretty_printer import prettyprint
from codegen.visitors.vulkan_kernel_visitor import VulkanKernelVisitor
from codegen.visitors.vulkan_cpp_stub_visitor import VulkanCppStubVisitor

log = logging.getLogger(__name__)


def read_file(filename: str) -> str:
    with open(filename) as f:
        return f.read()


def compile(filename: str) -> Program:
    print(f"--------------- parsing: {filename} -----------------")
    text = read_file(filename)
    ret = parser.parse(text)
    program = transform(ret)
    return program


def generate_vulkan(filename: str, output: str) -> None:
    """Parse kernel file and generate a Vulkan GLSL compute shader."""

    program = compile(filename)
    visitor = VulkanKernelVisitor()
    shader = program.accept(visitor)
    with open(output, "w") as f:
        f.write(shader)
    print(f"Generated Vulkan shader -> {output}")


def compile_vulkan(input_file: str, output_spv: str) -> None:
    """Generate a Vulkan shader file and compile it to SPIR-V.

    Note: compilation requires standard GLSL-compatible types. Domain-specific
    types (e.g., flexible_rows_matrix) produce valid WGSL but may fail glslc
    unless appropriate struct definitions are provided.
    """
    # Generate intermediate GLSL file
    glsl_path = input_file.rsplit(".", 1)[0] + ".glsl"
    generate_vulkan(input_file, glsl_path)

    # Compile with glslc
    cmd = [
        "glslc",
        "-fshader-stage=compute",
        "-o",
        output_spv,
        "--target-env=vulkan1.2",
        glsl_path,
    ]

    print(f"running: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("Compilation failed:")
        print(result.stderr, file=_sys_mod.stderr)
        _sys_mod.exit(1)

    print(f"Compiled SPIR-V -> {output_spv}")


def generate_cpp_stub(filename: str, output: str) -> None:
    """Parse kernel file and generate a C++ stub for calling the Vulkan kernel."""

    program = compile(filename)
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
    args = _parser.parse_args()

    if args.vulkan or args.compile:
        if not args.file:
            _parser.error("--vulkan and --compile require an input FILE argument")
        if args.vulkan:
            generate_vulkan(args.file, args.vulkan)
        elif args.compile:
            compile_vulkan(args.file, args.compile)
    elif args.cpp_stub:
        if not args.file:
            _parser.error("--cpp-stub requires an input FILE argument")
        generate_cpp_stub(args.file, args.cpp_stub)
    else:
        # Default: prettyprint all files
        for path in _sys_mod.argv[1:]:
            program = compile(path)
            print(prettyprint(program))
