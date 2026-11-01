/** @file test_hip_stubs.cc
 *  Unit tests that call all generated C++ stub dispatch functions using
 *  real Hip compute shaders, buffer allocations, and descriptor sets —
 *  with actual numerical verification of kernel outputs.
 */

#include <cstring>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

// Shared Hip test infrastructure
#include "hip_test_helpers.hpp"
#include "vecmath.L231.h"
#include "vecmath.L231.noopt.h"
#include "vecmath.L304.h"
#include "vecmath.L304.noopt.h"
#include "vecmath.L999.h"
#include "vecmath.L999.blas.h"
#include "vecmath.L999.noopt.h"


// ───────────── Test: multi-arg (3 independent matmuls + accumulate) ──


#if 0
static void write_multi_arg_descriptors(
    HipComputeKernel& kernel,
    VkDevice device,
    const std::vector<VBaseDeviceBuffer*>& bufs)
{
    std::vector<VkDescriptorBufferInfo> dbis(bufs.size());
    for (uint32_t i = 0; i < bufs.size(); ++i) {
        dbis[i].buffer = bufs[i]->get();
        dbis[i].offset = 0;
        dbis[i].range  = VK_WHOLE_SIZE;

        VkWriteDescriptorSet wds{};
        wds.sType             = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wds.dstSet            = kernel.desc_set();
        wds.dstBinding        = i;
        wds.descriptorCount   = 1;
        wds.descriptorType    = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wds.pBufferInfo       = &dbis[i];
        vkUpdateDescriptorSets(device, 1, &wds, 0, nullptr);
    }
}
#endif


template <typename T>
static std::vector<float> read_float_buffer(HipComputeContext& context, VDeviceBuffer<T>& buf, VHostBuffer<T>& host)
{
    const size_t size_bytes = host.size();
    std::vector<float> actual(static_cast<size_t>(size_bytes / sizeof(float)));
    buf.read(context, host);
    memcpy(actual.data(), host.get(), static_cast<size_t>(size_bytes));
    return actual;
}

#if 0
static bool selected_device_is_llvmpipe(HipSession& session)
{
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(session.get_phys_device(), &props);
    return strstr(props.deviceName, "llvmpipe") != nullptr;
}

static bool selected_device_is_dzn(HipSession& session)
{
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(session.get_phys_device(), &props);
    return strstr(props.deviceName, "dzn") != nullptr ||
           strstr(props.deviceName, "Direct3D12") != nullptr;
}

static std::string read_text_file(const std::string& path)
{
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("failed to open " + path);
    }
    return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}
#endif

TEST_F(HipTestBase, host_device_buffer_copy_bandwidth)
{
    constexpr size_t size_bytes = 64ull * 1024ull * 1024ull;
    constexpr int iterations = 5;

    VHostBuffer<uint8_t[size_bytes]> upload(get_session());
    VHostBuffer<uint8_t[size_bytes]> download(get_session());
    VDeviceBuffer<uint8_t[size_bytes]> device(upload);
    HipComputeContext copy_context(get_session());
    HipQueue& queue = get_session().get_queue(0);

    for (size_t i = 0; i < size_bytes; ++i) {
        upload.get()[i] = static_cast<uint8_t>((i * 131u + 17u) & 0xffu);
    }
    std::memset(download.get(), 0, static_cast<size_t>(size_bytes));

    device.write(queue, upload);
    device.read(queue, download);
    ASSERT_EQ(std::memcmp(upload.get(), download.get(), static_cast<size_t>(size_bytes)), 0);

    double upload_ms = 1.0e30;
    double download_ms = 1.0e30;
    double roundtrip_ms = 1.0e30;

    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::steady_clock::now();
        device.write(queue, upload);
        auto mid = std::chrono::steady_clock::now();
        device.read(queue, download);
        auto end = std::chrono::steady_clock::now();

        upload_ms = std::min(upload_ms, std::chrono::duration<double, std::milli>(mid - start).count());
        download_ms = std::min(download_ms, std::chrono::duration<double, std::milli>(end - mid).count());
        roundtrip_ms = std::min(roundtrip_ms, std::chrono::duration<double, std::milli>(end - start).count());
    }

    ASSERT_EQ(std::memcmp(upload.get(), download.get(), static_cast<size_t>(size_bytes)), 0);

    const double gib = static_cast<double>(size_bytes) / (1024.0 * 1024.0 * 1024.0);
    const double upload_gibs = gib / (upload_ms / 1000.0);
    const double download_gibs = gib / (download_ms / 1000.0);
    const double roundtrip_gibs = (2.0 * gib) / (roundtrip_ms / 1000.0);

    RecordProperty("size_bytes", static_cast<long long>(size_bytes));
    RecordProperty("upload_gibs", upload_gibs);
    RecordProperty("download_gibs", download_gibs);
    RecordProperty("roundtrip_gibs", roundtrip_gibs);
    std::cerr << "host/device buffer copy bandwidth: "
              << "upload=" << upload_gibs << " GiB/s"
              << ", download=" << download_gibs << " GiB/s"
              << ", roundtrip=" << roundtrip_gibs << " GiB/s"
              << " (" << size_bytes << " bytes, best of " << iterations << ")\n";
}

TEST_F(HipTestBase, ffn_act_optimized_and_blas_clip_match_sequential_loop)
{
    namespace optimized_kernel = rllm_offload_parfor_vecmath_999;
    namespace blas_kernel = rllm_offload_parfor_vecmath_999_blas;
    namespace sequential_kernel = rllm_offload_parfor_vecmath_999_noopt;

    constexpr uint32_t rows = 4;
    constexpr uint32_t cols = optimized_kernel::B_X;
    constexpr uint32_t inner = optimized_kernel::A_Y;

    ASSERT_STREQ(
        blas_kernel::OffloadParforVecmath999BlasVecmathKernel::generated_descriptor(),
        "backend=hipblas;operation=gemm;transpose_a=no;transpose_b=yes;beta=0;input=float16");
    ASSERT_EQ(
        std::string(optimized_kernel::OffloadParforVecmath999VecmathKernel::generated_descriptor())
            .find("backend=hipblas"),
        std::string::npos);

    VHostBuffer<optimized_kernel::RllmBuffer_A> host_a(get_session());
    VHostBuffer<optimized_kernel::RllmBuffer_B> host_b(get_session());
    VHostBuffer<optimized_kernel::RllmBuffer_C> host_optimized(get_session());
    VHostBuffer<blas_kernel::RllmBuffer_C> host_blas(get_session());
    VHostBuffer<sequential_kernel::RllmBuffer_C> host_sequential(get_session());

    // Use the production FFN matrix dimensions. Integral values are exactly
    // representable in fp16; rows 1 and 2 exercise both clipping branches.
    std::memset(host_a.get(), 0, static_cast<size_t>(host_a.size()));
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t k = 0; k < inner; ++k) {
            const float alternating = ((i + k) & 1u) ? -1.0f : 1.0f;
            const float value = i == 1 ? 100.0f : (i == 2 ? -100.0f : alternating);
            host_a.get()->data[i * inner + k] = static_cast<float16_t>(value);
        }
    }
    for (uint32_t j = 0; j < cols; ++j) {
        for (uint32_t k = 0; k < inner; ++k) {
            const float alternating = ((j + k) & 1u) ? -1.0f : 1.0f;
            const float value = j == 1 ? 100.0f : (j == 2 ? -100.0f : alternating);
            host_b.get()->data[j * inner + k] = static_cast<float16_t>(value);
        }
    }
    std::memset(host_optimized.get(), 0, static_cast<size_t>(host_optimized.size()));
    std::memset(host_blas.get(), 0, static_cast<size_t>(host_blas.size()));
    std::memset(host_sequential.get(), 0, static_cast<size_t>(host_sequential.size()));

    VDeviceBuffer<optimized_kernel::RllmBuffer_A> device_a(host_a);
    VDeviceBuffer<optimized_kernel::RllmBuffer_B> device_b(host_b);
    VDeviceBuffer<optimized_kernel::RllmBuffer_C> device_optimized(host_optimized);
    VDeviceBuffer<blas_kernel::RllmBuffer_C> device_blas(host_blas);
    VDeviceBuffer<sequential_kernel::RllmBuffer_C> device_sequential(host_sequential);
    HipQueue& queue = get_session().get_queue(0);
    device_a.write(queue, host_a);
    device_b.write(queue, host_b);
    device_optimized.write(queue, host_optimized);
    device_blas.write(queue, host_blas);
    device_sequential.write(queue, host_sequential);

    optimized_kernel::OffloadParforVecmath999VecmathKernel optimized(
        get_session(), std::string(TESTDATA_DIR) + "/vecmath.L999.hsaco");
    blas_kernel::OffloadParforVecmath999BlasVecmathKernel blas(
        get_session(), std::string(TESTDATA_DIR) + "/vecmath.L999.blas.hsaco");
    sequential_kernel::OffloadParforVecmath999NooptVecmathKernel sequential(
        get_session(), std::string(TESTDATA_DIR) + "/vecmath.L999.noopt.hsaco");
    const optimized_kernel::offload_parfor_vecmath_999_vecmath_PushConstants optimized_pc{
        static_cast<int32_t>(rows), static_cast<int32_t>(cols)};
    const blas_kernel::offload_parfor_vecmath_999_blas_vecmath_PushConstants blas_pc{
        static_cast<int32_t>(rows), static_cast<int32_t>(cols)};
    const sequential_kernel::offload_parfor_vecmath_999_noopt_vecmath_PushConstants sequential_pc{
        static_cast<int32_t>(rows), static_cast<int32_t>(cols)};

    sequential.dispatch(queue, rows, cols, device_a, device_b, device_sequential, sequential_pc);
    optimized.dispatch(queue, rows, cols, device_a, device_b, device_optimized, optimized_pc);
    blas.dispatch(queue, rows, cols, device_a, device_b, device_blas, blas_pc);
    device_sequential.read(queue, host_sequential);
    device_optimized.read(queue, host_optimized);
    device_blas.read(queue, host_blas);

    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            const uint32_t index = i * cols + j;
            EXPECT_FLOAT_EQ(host_optimized.get()->data[index], host_sequential.get()->data[index])
                << "optimized output[" << i << ", " << j << "]";
            EXPECT_FLOAT_EQ(host_blas.get()->data[index], host_sequential.get()->data[index])
                << "BLAS output[" << i << ", " << j << "]";
        }
    }
    EXPECT_FLOAT_EQ(host_blas.get()->data[1 * cols + 1], 10000.0f);
    EXPECT_FLOAT_EQ(host_blas.get()->data[1 * cols + 2], -10000.0f);
}

TEST_F(HipTestBase, tree_rewritten_vecmath231_matches_sequential_cpu)
{
    namespace kernel = rllm_offload_parfor_vecmath_231;

    constexpr uint32_t rows = 8;
    constexpr uint32_t cols = 8;
    constexpr uint32_t k_count = kernel::A_Y;

    ASSERT_STREQ(
        kernel::OffloadParforVecmath231VecmathKernel::generated_descriptor(),
        "tile_block_size=512;tile_size_x=8;tile_size_y=8;tile_chunk_size=8;workgroup_count=1;"
        "workgroup_size_x=8;workgroup_size_y=8;workgroup_size_z=8;reduction_chunks=8;num_z_threads=1;"
        "tree_transformed=yes;");

    VHostBuffer<kernel::RllmBuffer_A> host_a(get_session());
    VHostBuffer<kernel::RllmBuffer_B1> host_b1(get_session());
    VHostBuffer<kernel::RllmBuffer_C1> host_c1(get_session());
    VHostBuffer<kernel::RllmBuffer_B2> host_b2(get_session());
    VHostBuffer<kernel::RllmBuffer_C2> host_c2(get_session());
    VHostBuffer<kernel::RllmBuffer_B3> host_b3(get_session());
    VHostBuffer<kernel::RllmBuffer_C3> host_c3(get_session());

    for (uint32_t idx = 0; idx < kernel::A_X * kernel::A_Y; ++idx) {
        host_a.get()->data[idx] = 0.0f;
        host_c1.get()->data[idx] = 0.0f;
        host_c2.get()->data[idx] = 0.0f;
        host_c3.get()->data[idx] = 0.0f;
    }
    for (uint32_t idx = 0; idx < kernel::B1_X * kernel::B1_Y; ++idx) {
        host_b1.get()->data[idx] = static_cast<float16_t>(0.0f);
        host_b2.get()->data[idx] = static_cast<float16_t>(0.0f);
        host_b3.get()->data[idx] = static_cast<float16_t>(0.0f);
    }

    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t k = 0; k < k_count; ++k) {
            host_a.get()->data[kernel::A_Y * i + k] =
                0.01f * static_cast<float>(i + 1) + 0.001f * static_cast<float>(static_cast<int>(k % 17) - 8);
        }
    }
    for (uint32_t j = 0; j < cols; ++j) {
        for (uint32_t k = 0; k < k_count; ++k) {
            const uint32_t idx = kernel::B1_Y * j + k;
            host_b1.get()->data[idx] =
                static_cast<float16_t>(0.02f * static_cast<float>(j + 1) + 0.0005f * static_cast<float>(k % 11));
            host_b2.get()->data[idx] =
                static_cast<float16_t>(-0.015f * static_cast<float>(j + 1) + 0.0007f * static_cast<float>(k % 13));
            host_b3.get()->data[idx] =
                static_cast<float16_t>(0.01f * static_cast<float>(j + 2) - 0.0003f * static_cast<float>(k % 7));
        }
    }

    VDeviceBuffer<kernel::RllmBuffer_A> device_a(host_a);
    VDeviceBuffer<kernel::RllmBuffer_B1> device_b1(host_b1);
    VDeviceBuffer<kernel::RllmBuffer_C1> device_c1(host_c1);
    VDeviceBuffer<kernel::RllmBuffer_B2> device_b2(host_b2);
    VDeviceBuffer<kernel::RllmBuffer_C2> device_c2(host_c2);
    VDeviceBuffer<kernel::RllmBuffer_B3> device_b3(host_b3);
    VDeviceBuffer<kernel::RllmBuffer_C3> device_c3(host_c3);

    HipComputeContext context(get_session());
    auto& queue = get_session().get_queue(0);
    device_a.write(queue, host_a);
    device_b1.write(queue, host_b1);
    device_c1.write(queue, host_c1);
    device_b2.write(queue, host_b2);
    device_c2.write(queue, host_c2);
    device_b3.write(queue, host_b3);
    device_c3.write(queue, host_c3);

    kernel::OffloadParforVecmath231VecmathKernel vecmath231(
        get_session(),
        std::string(TESTDATA_DIR) + "/vecmath.L231.hsaco");
    const kernel::offload_parfor_vecmath_231_vecmath_PushConstants push_constants{
        static_cast<int32_t>(rows),
        static_cast<int32_t>(cols),
    };

    vecmath231.dispatch(
        queue,
        rows,
        cols,
        device_a,
        device_b1,
        device_c1,
        device_b2,
        device_c2,
        device_b3,
        device_c3,
        push_constants);

    device_c1.read(queue, host_c1);
    device_c2.read(queue, host_c2);
    device_c3.read(queue, host_c3);

    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            float expected1 = 0.0f;
            float expected2 = 0.0f;
            float expected3 = 0.0f;
            for (uint32_t k = 0; k < k_count; ++k) {
                const float a = host_a.get()->data[kernel::A_Y * i + k];
                expected1 += a * static_cast<float>(host_b1.get()->data[kernel::B1_Y * j + k]);
                expected2 += a * static_cast<float>(host_b2.get()->data[kernel::B2_Y * j + k]);
                expected3 += a * static_cast<float>(host_b3.get()->data[kernel::B3_Y * j + k]);
            }

            const uint32_t out_idx = kernel::C1_Y * i + j;
            EXPECT_NEAR(host_c1.get()->data[out_idx], expected1, 1.0e-2f) << "C1[" << i << ", " << j << "]";
            EXPECT_NEAR(host_c2.get()->data[out_idx], expected2, 1.0e-2f) << "C2[" << i << ", " << j << "]";
            EXPECT_NEAR(host_c3.get()->data[out_idx], expected3, 1.0e-2f) << "C3[" << i << ", " << j << "]";
        }
    }
}

TEST_F(HipTestBase, tree_rewritten_vecmath231_dispatch_performance)
{
    namespace kernel = rllm_offload_parfor_vecmath_231;
    namespace baseline_kernel = rllm_offload_parfor_vecmath_231_noopt;

    constexpr uint32_t rows = 64;
    constexpr uint32_t cols = 64;
    constexpr uint32_t k_count = kernel::A_Y;
    constexpr int warmup_iterations = 2;
    constexpr int measured_iterations = 10;

    VHostBuffer<kernel::RllmBuffer_A> host_a(get_session());
    VHostBuffer<kernel::RllmBuffer_B1> host_b1(get_session());
    VHostBuffer<kernel::RllmBuffer_C1> host_c1(get_session());
    VHostBuffer<kernel::RllmBuffer_B2> host_b2(get_session());
    VHostBuffer<kernel::RllmBuffer_C2> host_c2(get_session());
    VHostBuffer<kernel::RllmBuffer_B3> host_b3(get_session());
    VHostBuffer<kernel::RllmBuffer_C3> host_c3(get_session());

    std::memset(host_a.get(), 0, static_cast<size_t>(host_a.size()));
    std::memset(host_b1.get(), 0, static_cast<size_t>(host_b1.size()));
    std::memset(host_b2.get(), 0, static_cast<size_t>(host_b2.size()));
    std::memset(host_b3.get(), 0, static_cast<size_t>(host_b3.size()));

    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t k = 0; k < k_count; ++k) {
            host_a.get()->data[kernel::A_Y * i + k] =
                0.01f * static_cast<float>(i + 1) + 0.001f * static_cast<float>(static_cast<int>(k % 17) - 8);
        }
    }
    for (uint32_t j = 0; j < cols; ++j) {
        for (uint32_t k = 0; k < k_count; ++k) {
            const uint32_t idx = kernel::B1_Y * j + k;
            host_b1.get()->data[idx] =
                static_cast<float16_t>(0.02f * static_cast<float>(j + 1) + 0.0005f * static_cast<float>(k % 11));
            host_b2.get()->data[idx] =
                static_cast<float16_t>(-0.015f * static_cast<float>(j + 1) + 0.0007f * static_cast<float>(k % 13));
            host_b3.get()->data[idx] =
                static_cast<float16_t>(0.01f * static_cast<float>(j + 2) - 0.0003f * static_cast<float>(k % 7));
        }
    }

    VDeviceBuffer<kernel::RllmBuffer_A> device_a(host_a);
    VDeviceBuffer<kernel::RllmBuffer_B1> device_b1(host_b1);
    VDeviceBuffer<kernel::RllmBuffer_C1> device_c1(host_c1);
    VDeviceBuffer<kernel::RllmBuffer_B2> device_b2(host_b2);
    VDeviceBuffer<kernel::RllmBuffer_C2> device_c2(host_c2);
    VDeviceBuffer<kernel::RllmBuffer_B3> device_b3(host_b3);
    VDeviceBuffer<kernel::RllmBuffer_C3> device_c3(host_c3);

    HipComputeContext context(get_session());
    HipQueue& queue = get_session().get_queue(0);

    device_a.write(queue, host_a);
    device_b1.write(queue, host_b1);
    device_b2.write(queue, host_b2);
    device_b3.write(queue, host_b3);

    kernel::OffloadParforVecmath231VecmathKernel vecmath231(
        get_session(),
        std::string(TESTDATA_DIR) + "/vecmath.L231.hsaco");
    const kernel::offload_parfor_vecmath_231_vecmath_PushConstants push_constants{
        static_cast<int32_t>(rows),
        static_cast<int32_t>(cols),
    };
    baseline_kernel::OffloadParforVecmath231NooptVecmathKernel baseline_vecmath231(
        get_session(),
        std::string(TESTDATA_DIR) + "/vecmath.L231.noopt.hsaco");
    const baseline_kernel::offload_parfor_vecmath_231_noopt_vecmath_PushConstants baseline_push_constants{
        static_cast<int32_t>(rows),
        static_cast<int32_t>(cols),
    };

    auto dispatch_transformed_once = [&]() {
        const auto start = std::chrono::steady_clock::now();
        vecmath231.dispatch(
            queue,
            rows,
            cols,
            device_a,
            device_b1,
            device_c1,
            device_b2,
            device_c2,
            device_b3,
            device_c3,
            push_constants);
        const auto end = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    };
    auto dispatch_baseline_once = [&]() {
        const auto start = std::chrono::steady_clock::now();
        baseline_vecmath231.dispatch(
            queue,
            rows,
            cols,
            device_a,
            device_b1,
            device_c1,
            device_b2,
            device_c2,
            device_b3,
            device_c3,
            baseline_push_constants);
        const auto end = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    };

    for (int i = 0; i < warmup_iterations; ++i) {
        (void)dispatch_baseline_once();
        (void)dispatch_transformed_once();
    }

    double best_transformed_ms = 1.0e30;
    double total_transformed_ms = 0.0;
    double best_baseline_ms = 1.0e30;
    double total_baseline_ms = 0.0;
    for (int i = 0; i < measured_iterations; ++i) {
        const double baseline_ms = dispatch_baseline_once();
        const double transformed_ms = dispatch_transformed_once();
        best_baseline_ms = std::min(best_baseline_ms, baseline_ms);
        total_baseline_ms += baseline_ms;
        best_transformed_ms = std::min(best_transformed_ms, transformed_ms);
        total_transformed_ms += transformed_ms;
    }

    const double average_transformed_ms = total_transformed_ms / static_cast<double>(measured_iterations);
    const double average_baseline_ms = total_baseline_ms / static_cast<double>(measured_iterations);
    const double operations = static_cast<double>(rows) * static_cast<double>(cols) *
                              static_cast<double>(k_count) * 3.0 * 2.0;
    const double best_transformed_gflops = operations / (best_transformed_ms / 1000.0) / 1.0e9;
    const double average_transformed_gflops = operations / (average_transformed_ms / 1000.0) / 1.0e9;
    const double best_baseline_gflops = operations / (best_baseline_ms / 1000.0) / 1.0e9;
    const double average_baseline_gflops = operations / (average_baseline_ms / 1000.0) / 1.0e9;
    const double best_speedup = best_baseline_ms / best_transformed_ms;
    const double average_speedup = average_baseline_ms / average_transformed_ms;

    ASSERT_GT(best_baseline_ms, 0.0);
    ASSERT_GT(best_transformed_ms, 0.0);

    RecordProperty("rows", rows);
    RecordProperty("cols", cols);
    RecordProperty("k_count", k_count);
    RecordProperty("iterations", measured_iterations);
    RecordProperty("baseline_best_ms", best_baseline_ms);
    RecordProperty("baseline_average_ms", average_baseline_ms);
    RecordProperty("baseline_best_gflops", best_baseline_gflops);
    RecordProperty("baseline_average_gflops", average_baseline_gflops);
    RecordProperty("transformed_best_ms", best_transformed_ms);
    RecordProperty("transformed_average_ms", average_transformed_ms);
    RecordProperty("transformed_best_gflops", best_transformed_gflops);
    RecordProperty("transformed_average_gflops", average_transformed_gflops);
    RecordProperty("best_speedup", best_speedup);
    RecordProperty("average_speedup", average_speedup);
    std::cerr << "vecmath:231 dispatch performance: "
              << "rows=" << rows
              << ", cols=" << cols
              << ", k=" << k_count
              << ", baseline_best=" << best_baseline_ms << " ms"
              << ", transformed_best=" << best_transformed_ms << " ms"
              << ", best_speedup=" << best_speedup << "x"
              << ", baseline_avg=" << average_baseline_ms << " ms"
              << ", transformed_avg=" << average_transformed_ms << " ms"
              << ", avg_speedup=" << average_speedup << "x"
              << ", baseline_best=" << best_baseline_gflops << " GFLOP/s"
              << ", transformed_best=" << best_transformed_gflops << " GFLOP/s"
              << " (" << measured_iterations << " measured iterations)\n";
}

TEST_F(HipTestBase, tree_rewritten_vecmath304_tiled_shared_cache_performance)
{
    namespace kernel = rllm_offload_parfor_vecmath_304;
    namespace baseline_kernel = rllm_offload_parfor_vecmath_304_noopt;

    constexpr uint32_t rows = 64;
    constexpr uint32_t cols = 64;
    constexpr uint32_t k_count = kernel::A_Y;
    constexpr int warmup_iterations = 2;
    constexpr int measured_iterations = 10;

    VHostBuffer<kernel::RllmBuffer_A> host_a(get_session());
    VHostBuffer<kernel::RllmBuffer_B> host_b(get_session());
    VHostBuffer<kernel::RllmBuffer_C> host_c(get_session());
    VHostBuffer<kernel::RllmBuffer_C> host_c_baseline(get_session());
    VHostBuffer<kernel::RllmBuffer_C> host_c_transformed(get_session());

    std::memset(host_a.get(), 0, static_cast<size_t>(host_a.size()));
    std::memset(host_b.get(), 0, static_cast<size_t>(host_b.size()));
    std::memset(host_c.get(), 0, static_cast<size_t>(host_c.size()));

    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t k = 0; k < k_count; ++k) {
            host_a.get()->data[kernel::A_Y * i + k] =
                0.01f * static_cast<float>(i + 1) + 0.001f * static_cast<float>(static_cast<int>(k % 17) - 8);
        }
    }
    for (uint32_t j = 0; j < cols; ++j) {
        for (uint32_t k = 0; k < k_count; ++k) {
            host_b.get()->data[kernel::B_Y * j + k] =
                static_cast<float16_t>(0.02f * static_cast<float>(j + 1) + 0.0005f * static_cast<float>(k % 11));
        }
    }

    VDeviceBuffer<kernel::RllmBuffer_A> device_a(host_a);
    VDeviceBuffer<kernel::RllmBuffer_B> device_b(host_b);
    VDeviceBuffer<kernel::RllmBuffer_C> device_c(host_c);
    VDeviceBuffer<kernel::RllmBuffer_C> device_c_baseline(host_c_baseline);
    VDeviceBuffer<kernel::RllmBuffer_C> device_c_transformed(host_c_transformed);

    std::memset(host_c_baseline.get(), 0, static_cast<size_t>(host_c_baseline.size()));
    std::memset(host_c_transformed.get(), 0, static_cast<size_t>(host_c_transformed.size()));

    HipComputeContext context(get_session());
    auto& queue = get_session().get_queue(0);
    device_a.write(queue, host_a);
    device_b.write(queue, host_b);

    kernel::OffloadParforVecmath304VecmathKernel vecmath304(
        get_session(),
        std::string(TESTDATA_DIR) + "/vecmath.L304.hsaco");
    const kernel::offload_parfor_vecmath_304_vecmath_PushConstants push_constants{
        static_cast<int32_t>(rows),
        static_cast<int32_t>(cols),
    };
    baseline_kernel::OffloadParforVecmath304NooptVecmathKernel baseline_vecmath304(
        get_session(),
        std::string(TESTDATA_DIR) + "/vecmath.L304.noopt.hsaco");
    const baseline_kernel::offload_parfor_vecmath_304_noopt_vecmath_PushConstants baseline_push_constants{
        static_cast<int32_t>(rows),
        static_cast<int32_t>(cols),
    };

    auto dispatch_transformed_once = [&]() {
        const auto start = std::chrono::steady_clock::now();
        vecmath304.dispatch(
            queue,
            rows,
            cols,
            device_a,
            device_b,
            device_c,
            push_constants);
        const auto end = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    };
    auto dispatch_baseline_once = [&]() {
        const auto start = std::chrono::steady_clock::now();
        baseline_vecmath304.dispatch(
            queue,
            rows,
            cols,
            device_a,
            device_b,
            device_c,
            baseline_push_constants);
        const auto end = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    };

    for (int i = 0; i < warmup_iterations; ++i) {
        (void)dispatch_baseline_once();
        (void)dispatch_transformed_once();
    }

    double best_transformed_ms = 1.0e30;
    double total_transformed_ms = 0.0;
    double best_baseline_ms = 1.0e30;
    double total_baseline_ms = 0.0;
    for (int i = 0; i < measured_iterations; ++i) {
        const double baseline_ms = dispatch_baseline_once();
        const double transformed_ms = dispatch_transformed_once();
        best_baseline_ms = std::min(best_baseline_ms, baseline_ms);
        total_baseline_ms += baseline_ms;
        best_transformed_ms = std::min(best_transformed_ms, transformed_ms);
        total_transformed_ms += transformed_ms;
    }

    const double average_transformed_ms = total_transformed_ms / static_cast<double>(measured_iterations);
    const double average_baseline_ms = total_baseline_ms / static_cast<double>(measured_iterations);
    const double operations = static_cast<double>(rows) * static_cast<double>(cols) *
                              static_cast<double>(k_count) * 2.0;
    const double best_transformed_gflops = operations / (best_transformed_ms / 1000.0) / 1.0e9;
    const double average_transformed_gflops = operations / (average_transformed_ms / 1000.0) / 1.0e9;
    const double best_baseline_gflops = operations / (best_baseline_ms / 1000.0) / 1.0e9;
    const double average_baseline_gflops = operations / (average_baseline_ms / 1000.0) / 1.0e9;
    const double best_speedup = best_baseline_ms / best_transformed_ms;
    const double average_speedup = average_baseline_ms / average_transformed_ms;

    ASSERT_GT(best_baseline_ms, 0.0);
    ASSERT_GT(best_transformed_ms, 0.0);

    RecordProperty("rows", rows);
    RecordProperty("cols", cols);
    RecordProperty("k_count", k_count);
    RecordProperty("iterations", measured_iterations);
    RecordProperty("baseline_best_ms", best_baseline_ms);
    RecordProperty("baseline_average_ms", average_baseline_ms);
    RecordProperty("baseline_best_gflops", best_baseline_gflops);
    RecordProperty("baseline_average_gflops", average_baseline_gflops);
    RecordProperty("transformed_best_ms", best_transformed_ms);
    RecordProperty("transformed_average_ms", average_transformed_ms);
    RecordProperty("transformed_best_gflops", best_transformed_gflops);
    RecordProperty("transformed_average_gflops", average_transformed_gflops);
    RecordProperty("best_speedup", best_speedup);
    RecordProperty("average_speedup", average_speedup);
    std::cerr << "vecmath:304 tiled shared-cache performance: "
              << "rows=" << rows
              << ", cols=" << cols
              << ", k=" << k_count
              << ", baseline_best=" << best_baseline_ms << " ms"
              << ", transformed_best=" << best_transformed_ms << " ms"
              << ", best_speedup=" << best_speedup << "x"
              << ", baseline_avg=" << average_baseline_ms << " ms"
              << ", transformed_avg=" << average_transformed_ms << " ms"
              << ", avg_speedup=" << average_speedup << "x"
              << ", baseline_best=" << best_baseline_gflops << " GFLOP/s"
              << ", transformed_best=" << best_transformed_gflops << " GFLOP/s"
              << " (" << measured_iterations << " measured iterations)\n";
    // Correctness: dispatch both kernels once to separate buffers and compare
    device_c_baseline.write(queue, host_c_baseline);
    device_c_transformed.write(queue, host_c_transformed);
    baseline_vecmath304.dispatch(queue, rows, cols, device_a, device_b, device_c_baseline, baseline_push_constants);
    vecmath304.dispatch(queue, rows, cols, device_a, device_b, device_c_transformed, push_constants);
    device_c_baseline.read(queue, host_c_baseline);
    device_c_transformed.read(queue, host_c_transformed);

    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            const uint32_t idx = kernel::C_Y * i + j;
            const float baseline_val = host_c_baseline.get()->data[idx];
            const float transformed_val = host_c_transformed.get()->data[idx];
            EXPECT_NEAR(baseline_val, transformed_val, 1.0e-2f)
                << "Output mismatch at [" << i << ", " << j << "]: baseline=" << baseline_val
                << ", transformed=" << transformed_val;
        }
    }
}
