"""Unit tests for Vulkan compute shader C++ stub generation.

Each test:
1. Parses a .kernel file with the Lark-based parser
2. Generates a C++ stub header via VulkanCppStubVisitor
3. Wraps it in a small test harness that includes mock Vulkan types
4. Compiles the combined code with g++ to verify correctness
"""

import subprocess
import sys
import tempfile
import textwrap
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


KERNEL_DIR = _ROOT / "codegen" / "testdata"
MULTI_ARG_KERNEL = KERNEL_DIR / "multi-arg.kernel"
SINGLE_ASSIGN_KERNEL = KERNEL_DIR / "single-assign.kernel"


def _generate_and_compile(kernel_path: Path):
    """Parse the kernel, generate a stub header, wrap in mock Vulkan types,
    and attempt to compile. Returns (success: bool, stderr: str)."""
    from codegen.parser import parse
    from codegen.visitors.vulkan_cpp_stub_visitor import VulkanCppStubVisitor

    program = parse(str(kernel_path))
    visitor = VulkanCppStubVisitor()
    stub_content = program.accept(visitor)

    has_2d = program.space_dim >= 2 and len(program.loop_vars) >= 2

    # Classify params from the stub (parse buffer names from generated output)
    matrix_params = []
    pc_type = None

    for line in stub_content.splitlines():
        match = re.match(r"struct\s+(RllmBuffer_\w+)\s*\{", line)
        if match:
            matrix_params.append(match.group(1))

    pc_match = re.search(r"struct\s+(\w+_PushConstants)\s*\{", stub_content)
    if pc_match:
        pc_type = pc_match.group(1)

    # Mock Vulkan types
    mock_vulkan = textwrap.dedent("""\
        #include <cstdint>
        struct VkDevice_T{}; using VkDevice = VkDevice_T*;
        struct VkPipelineLayout_T{}; using VkPipelineLayout = VkPipelineLayout_T*;
        struct VkDescriptorSetLayout_T{}; using VkDescriptorSetLayout = VkDescriptorSetLayout_T*;
        struct VkCommandBuffer_T{}; using VkCommandBuffer = VkCommandBuffer_T*;
        struct VkDescriptorSet_T{}; using VkDescriptorSet = VkDescriptorSet_T*;
        inline void vkCmdDispatch(VkCommandBuffer, uint32_t, uint32_t, uint32_t){}
    """)

    harness_lines = [mock_vulkan]
    for line in stub_content.splitlines():
        if "#include <vulkan/vulkan.h>" in line:
            continue
        harness_lines.append(line)

    # Build main() with correct number of args matching the RLLM stub format
    init_code = []
    call_args = ["nullptr", "nullptr", "nullptr", "nullptr", "nullptr"]

    if has_2d:
        call_args.extend(["1u", "1u"])
    else:
        call_args.append("1u")

    for i, sname in enumerate(matrix_params):
        vname = f"_m{i}"
        init_code.append(f"    {sname} {vname};")
        call_args.append(vname)

    if pc_type:
        push_var_name = "_push"
        init_code.append(f"    const {pc_type} {push_var_name}{{}};")
        call_args.append(f"const_cast<{pc_type}&>({push_var_name})")

    # Find the dispatch function name from stub
    func_name = None
    for line in stub_content.splitlines():
        if "inline void" in line and "(" in line:
            func_part = line.replace("inline void ", "").strip().split("(")[0]
            func_name = func_part.rstrip(";{")
            break

    if not func_name:
        return False, "Could not find dispatch function name in stub"

    main_func = (
        "int main() {\n" + "\n".join(init_code) + "\n"
        f"    {func_name}({', '.join(call_args)});\n"
        "    return 0;\n"
        "}\n"
    )

    harness_lines.append(main_func)
    combined_source = "\n".join(harness_lines) + "\n"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        test_cpp = tmppath / "test_stub.cpp"
        test_cpp.write_text(combined_source)

        for compiler in ("g++", "clang++", "c++"):
            try:
                r = subprocess.run(
                    [
                        compiler,
                        "-std=c++17",
                        "-x",
                        "c++",
                        str(test_cpp),
                        "-o",
                        str(tmppath / "test_out"),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if r.returncode == 0:
                    return True, ""
            except FileNotFoundError:
                continue

        r = subprocess.run(
            [
                "g++",
                "-std=c++17",
                "-x",
                "c++",
                str(test_cpp),
                "-o",
                str(tmppath / "test_out"),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return False, r.stderr


# ---- tests ----


@pytest.mark.parametrize(
    "kernel_path",
    [
        MULTI_ARG_KERNEL,
        SINGLE_ASSIGN_KERNEL,
    ],
    ids=["multi-arg", "single-assign"],
)
def test_stub_compiles(kernel_path: Path):
    """Verify that the generated C++ stub compiles with mocked Vulkan types."""
    success, stderr = _generate_and_compile(kernel_path)
    assert success, (
        f"Generated C++ stub for {kernel_path.name} failed to compile.\n"
        f"stderr:\n{stderr}"
    )


@pytest.mark.parametrize(
    "kernel_path",
    [
        MULTI_ARG_KERNEL,
    ],
    ids=["multi-arg"],
)
def test_stub_has_vkcmddispatch_call(kernel_path: Path):
    """Verify the stub contains a vkCmdDispatch call."""
    from codegen.parser import parse
    from codegen.visitors.vulkan_cpp_stub_visitor import VulkanCppStubVisitor

    program = parse(str(kernel_path))
    visitor = VulkanCppStubVisitor()
    content = program.accept(visitor)
    assert "vkCmdDispatch" in content, "Missing vkCmdDispatch in generated stub"


@pytest.mark.parametrize(
    "kernel_path",
    [
        MULTI_ARG_KERNEL,
    ],
    ids=["multi-arg"],
)
def test_stub_multi_arg_dispatch_dimensions(kernel_path: Path):
    """Verify 2D parfor kernel dispatches with correct x/y workgroup division."""
    from codegen.parser import parse
    from codegen.visitors.vulkan_cpp_stub_visitor import VulkanCppStubVisitor

    program = parse(str(kernel_path))
    visitor = VulkanCppStubVisitor()
    content = program.accept(visitor)

    assert "dispatch_rows" in content, "Missing dispatch_rows param for 2D kernel"
    assert "dispatch_cols" in content, "Missing dispatch_cols param for 2D kernel"
    assert "(dispatch_rows + 8 - 1)" in content, "Missing workgroup X division"
    assert "(dispatch_cols + 8 - 1)" in content, "Missing workgroup Y division"


@pytest.mark.parametrize(
    "kernel_path",
    [
        MULTI_ARG_KERNEL,
        SINGLE_ASSIGN_KERNEL,
    ],
    ids=["multi-arg", "single-assign"],
)
def test_stub_has_buffer_struct(kernel_path: Path):
    """Verify stub has RllmBuffer struct definitions."""
    from codegen.parser import parse
    from codegen.visitors.vulkan_cpp_stub_visitor import VulkanCppStubVisitor

    program = parse(str(kernel_path))
    visitor = VulkanCppStubVisitor()
    content = program.accept(visitor)

    assert "struct RllmBuffer_" in content, "Missing RllmBuffer struct definitions"


@pytest.mark.parametrize(
    "kernel_path",
    [
        SINGLE_ASSIGN_KERNEL,
    ],
    ids=["single-assign"],
)
def test_stub_single_assign_has_push_constants(kernel_path: Path):
    """Verify single-assign kernel stub has push constants (it has scalar params)."""
    from codegen.parser import parse
    from codegen.visitors.vulkan_cpp_stub_visitor import VulkanCppStubVisitor

    program = parse(str(kernel_path))
    visitor = VulkanCppStubVisitor()
    content = program.accept(visitor)

    assert "_PushConstants" in content, (
        "Missing push constants struct for single-assign"
    )


@pytest.mark.parametrize(
    "kernel_path",
    [
        MULTI_ARG_KERNEL,
    ],
    ids=["multi-arg"],
)
def test_stub_multi_arg_has_all_matrix_buffers(kernel_path: Path):
    """Verify multi-arg kernel stub has all matrix buffer structs."""
    from codegen.parser import parse
    from codegen.visitors.vulkan_cpp_stub_visitor import VulkanCppStubVisitor

    program = parse(str(kernel_path))
    visitor = VulkanCppStubVisitor()
    content = program.accept(visitor)

    for name in ["A1", "B1", "A2", "B2", "A3", "B3", "C"]:
        assert f"RllmBuffer_{name}" in content, f"Missing RllmBuffer_{name} struct"
