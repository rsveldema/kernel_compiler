from unittest_offload_parfor_helper import parse_kernel


def test_parse():
    parse_kernel("offload_parfor_vecmath_193.cc.kernel")
