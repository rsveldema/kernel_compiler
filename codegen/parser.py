"""Compiler front-end: parse kernel files -> AST."""

import os
import sys as _sys_mod


def _fix_imports():
    """Prevent codegen/ast from shadowing stdlib ast when run as a script,
    and ensure package-level imports work correctly."""
    # When running as a script (python codegen/parser.py), Python adds the
    # directory of the script (/projects/rllm/codegen) to sys.path[0]. This
    # makes 'codegen/ast' shadow stdlib 'ast'. Fix by renaming the bad entry.
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if _script_dir in _sys_mod.path:
        bad_idx = _sys_mod.path.index(_script_dir)
        _sys_mod.path[bad_idx] = _script_dir + '_parser_tmp_shadow'

    # Ensure package root is on sys.path for relative imports when run as script
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if pkg_root not in _sys_mod.path:
        _sys_mod.path.insert(0, pkg_root)


_fix_imports()

from lark import Lark
from codegen.ast import Program
from codegen.visitors.pretty_printer import PrettyPrinter
from codegen.transforms import transform
import argparse
import subprocess


# Resolve grammar path relative to this file's directory
_GRAMMAR_PATH = os.path.join(os.path.dirname(__file__), "grammar.lark")
grammar = open(_GRAMMAR_PATH).read()
parser = Lark(grammar, start="program")


def read_file(filename: str) -> str:
    with open(filename) as f:
        return f.read()


def parse(filename: str):
    print(f"--------------- parsing: {filename} -----------------")
    text = read_file(filename)
    ret = parser.parse(text)
    program = transform(ret)
    return program


def prettyprint(program: Program):
    printer = PrettyPrinter()
    s = program.accept(printer)
    print(s)


def generate_vulkan(filename: str, output: str) -> None:
    """Parse kernel file and generate a Vulkan GLSL compute shader."""
    from codegen.visitors.vulkan_kernel_visitor import VulkanKernelVisitor

    program = parse(filename)
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
    result = subprocess.run(
        ["glslc", "-fshader-stage=compute", "-o", output_spv,
         "--target-env=vulkan1.2", glsl_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Compilation failed:")
        print(result.stderr, file=_sys_mod.stderr)
        _sys_mod.exit(1)

    print(f"Compiled SPIR-V -> {output_spv}")

def generate_cpp_stub(filename: str, output: str) -> None:
    """Parse kernel file and generate a C++ stub for calling the Vulkan kernel."""
    from codegen.visitors.vulkan_cpp_stub_visitor import VulkanCppStubVisitor

    program = parse(filename)
    visitor = VulkanCppStubVisitor()
    stub = program.accept(visitor)
    with open(output, "w") as f:
        f.write(stub)
    print(f"Generated C++ stub -> {output}")




if __name__ == "__main__":
    _parser = argparse.ArgumentParser(description="Kernel compiler front-end")
    _parser.add_argument("file", nargs="?", help="Input .kernel file (required for --vulkan/--compile)")
    _parser.add_argument("--vulkan", metavar="OUTPUT",
                         help="Generate Vulkan GLSL shader to OUTPUT")
    _parser.add_argument("--compile", metavar="OUTPUT_SPV",
                         help="Generate and compile Vulkan shader to SPIR-V")
    _parser.add_argument("--cpp-stub", metavar="OUTPUT_HPP",
                         help="Generate C++ stub header for kernel dispatch")
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
            program = parse(path)
            prettyprint(program)
