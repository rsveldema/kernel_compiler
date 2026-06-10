
Kernel Compiler
========================

The goal is the reduce the work needed to create optimized LLM implementations.

This repo contains a optimizing compiler that takes in .kernel files (the body of a 'for' loop) and produces Vulkan glsl files
that can be converted to spirv files to be executed on your nVidia/AMD/Intel/etc. GPU/CPU/accelerator.
Its main use is to have a simple way to generate optimized code from matrix-multiply/linear algebra fragments.
The generates code can then be called by your LLM implementation.
In the end the generated code should be performance wise close the BLAS (if the kernel is recognized to be compatible to 
one of the known BLAS structures, we could just call the BLAS function. This depends on the compile flags passed to the compiler).

For example, a matrix multiply is written naively using nested loops.
We then optimize the kernel using blocking, pipelining, multi-versioning, etc. to generate optimized versions
and then generate the glsl file.


Usage
-------------

```bash
python parser.py testdata/triangular1.kernel --compile output.spirv
```

This takes in a kernel and generates testdata/triangular1.glsl and then compiles it using glslc to create output.spirv.
To ease integration of the generated spirv files, we also generate a C++ stub to call the generated kernel from your application code.
