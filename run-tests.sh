#!/bin/sh

mkdir -p output

for i in testdata/*.kernel
do
	python -m codegen.compile $i --compile output/$i.spv
done

