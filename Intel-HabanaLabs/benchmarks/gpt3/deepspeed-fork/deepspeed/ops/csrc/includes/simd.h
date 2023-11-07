// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#if (__x86_64__ || __i386__)
#include <cpuid.h>
#include <x86intrin.h>
#endif
#include <cstdint>
#include <cstring>
#include <type_traits>

template <typename T>
inline T readAs(const void* src)
{
    T res;
    std::memcpy(&res, src, sizeof(T));
    return res;
}

template <typename T>
inline void writeAs(void* dst, const T& val)
{
    std::memcpy(dst, &val, sizeof(T));
}

#define TILE (128 * 1024 * 1024)
#if defined(__AVX512__) or defined(__AVX256__)

#define ROUND_DOWN(size, step) ((size) & ~((step)-1))

#if defined(__AVX512__)
#define SIMD_STORE(a, d) _mm512_storeu_ps(a, d)
#define SIMD_LOAD(x) _mm512_loadu_ps(x)
#define SIMD_SET(x) _mm512_set1_ps(x)
#define SIMD_ADD(x, y) _mm512_add_ps(x, y)
#define SIMD_MUL(x, y) _mm512_mul_ps(x, y)
#define SIMD_FMA(x, y, c) _mm512_fmadd_ps(x, y, c)
#define SIMD_SQRT(x) _mm512_sqrt_ps(x)
#define SIMD_DIV(x, y) _mm512_div_ps(x, y)
#define SIMD_WIDTH 16
static __m512 load_16_bf16_as_f32(const void* data)
{
    __m256i a = readAs<__m256i>(data);     // use memcpy to avoid aliasing
    __m512i b = _mm512_cvtepu16_epi32(a);  // convert 8 u16 to 8 u32
    __m512i c = _mm512_slli_epi32(b, 16);  // logical shift left of all u32 by
                                           // 16 bits (representing bf16->f32)
    return readAs<__m512>(&c);             // use memcpy to avoid aliasing
}

static void store_16_f32_as_bf16_nearest(__m512 v, void* data)
{
    __m512i u32 = readAs<__m512i>(&v);

    // flow assuming non-nan:

    // uint32_t rounding_bias = ((U32 >> 16) & 1) + UINT32_C(0x7FFF);
    __m512i b = _mm512_srli_epi32(u32, 16);
    __m512i lsb_mask = _mm512_set1_epi32(0x00000001);
    __m512i c = _mm512_and_si512(b, lsb_mask);
    __m512i bias_constant = _mm512_set1_epi32(0x00007fff);
    __m512i rounding_bias = _mm512_add_epi32(c, bias_constant);

    // uint16_t res = static_cast<uint16_t>((U32 + rounding_bias) >> 16);
    __m512i d = _mm512_add_epi32(u32, rounding_bias);
    __m512i e = _mm512_srli_epi32(d, 16);
    __m256i non_nan_res = _mm512_cvtusepi32_epi16(e);

    // handle nan (exp is all 1s and mantissa != 0)
    // if ((x & 0x7fffffffU) > 0x7f800000U)
    __m512i mask_out_sign = _mm512_set1_epi32(0x7fffffff);
    __m512i non_sign_bits = _mm512_and_si512(u32, mask_out_sign);
    __m512i nan_threshold = _mm512_set1_epi32(0x7f800000);
    __mmask16 nan_mask = _mm512_cmp_epi32_mask(non_sign_bits, nan_threshold, _MM_CMPINT_GT);

    // mix in results with nans as needed
    __m256i nans = _mm256_set1_epi16(0x7fc0);
    __m256i res = _mm256_mask_mov_epi16(non_nan_res, nan_mask, nans);

    writeAs(data, res);
}

#ifdef USE_HPU
#define SIMD_LOAD2(x, h) ((h) ? load_16_bf16_as_f32(x) : _mm512_loadu_ps(x))

#define SIMD_STORE2(x, d, h) ((h) ? store_16_f32_as_bf16_nearest(d, x) : _mm512_storeu_ps(x, d))
#else  // USE_HPU
#define SIMD_LOAD2(x, h) \
    ((h) ? _mm512_cvtph_ps(_mm256_castps_si256(_mm256_loadu_ps(x))) : _mm512_loadu_ps(x))
#define SIMD_STORE2(x, d, h)                                                                      \
    ((h) ? _mm256_store_ps(x, _mm256_castsi256_ps(_mm512_cvtps_ph(d, _MM_FROUND_TO_NEAREST_INT))) \
         : _mm512_storeu_ps(x, d))
#endif  // USE_HPU

#define INTV __m256i
#elif defined(__AVX256__)
#define SIMD_STORE(a, d) _mm256_storeu_ps(a, d)
#define SIMD_LOAD(x) _mm256_loadu_ps(x)
#define SIMD_SET(x) _mm256_set1_ps(x)
#define SIMD_ADD(x, y) _mm256_add_ps(x, y)
#define SIMD_MUL(x, y) _mm256_mul_ps(x, y)
#define SIMD_FMA(x, y, c) _mm256_fmadd_ps(x, y, c)
#define SIMD_SQRT(x) _mm256_sqrt_ps(x)
#define SIMD_DIV(x, y) _mm256_div_ps(x, y)
#define SIMD_WIDTH 8
__m256 load_8_bf16_as_f32(const float* data)
{
    __m128i a = readAs<__m128i>(data);     // use memcpy to avoid aliasing
    __m256i b = _mm256_cvtepu16_epi32(a);  // convert 8 u16 to 8 u32
    __m256i c = _mm256_slli_epi32(b, 16);  // logical shift left of all u32 by
                                           // 16 bits (representing bf16->f32)
    return readAs<__m256>(&c);             // use memcpy to avoid aliasing
}

void store_8_f32_as_bf16_nearest(__m256 v, float* data)
{
    __m256i u32 = readAs<__m256i>(&v);

    // flow assuming non-nan:

    // uint32_t rounding_bias = ((U32 >> 16) & 1) + UINT32_C(0x7FFF);
    __m256i b = _mm256_srli_epi32(u32, 16);
    __m256i lsb_mask = _mm256_set1_epi32(0x00000001);
    __m256i c = _mm256_and_si256(b, lsb_mask);
    __m256i bias_constant = _mm256_set1_epi32(0x00007fff);
    __m256i rounding_bias = _mm256_add_epi32(c, bias_constant);

    // uint16_t res = static_cast<uint16_t>((U32 + rounding_bias) >> 16);
    __m256i d = _mm256_add_epi32(u32, rounding_bias);
    __m256i e = _mm256_srli_epi32(d, 16);
    __m128i non_nan_res = _mm256_cvtusepi32_epi16(e);

    // handle nan (exp is all 1s and mantissa != 0)
    // if ((x & 0x7fffffffU) > 0x7f800000U)
    __m256i mask_out_sign = _mm256_set1_epi32(0x7fffffff);
    __m256i non_sign_bits = _mm256_and_si256(u32, mask_out_sign);
    __m256i nan_threshold = _mm256_set1_epi32(0x7f800000);
    __mmask8 nan_mask = _mm256_cmp_epi32_mask(non_sign_bits, nan_threshold, _MM_CMPINT_GT);

    // mix in results with nans as needed
    __m128i nans = _mm_set1_epi16(0x7fc0);
    __m128i res = _mm_mask_mov_epi16(non_nan_res, nan_mask, nans);

    writeAs(data, res);
}
#if defined(USE_HPU)
#define SIMD_LOAD2(x, h) ((h) ? load_8_bf16_as_f32(x) : _mm256_loadu_ps(x))

#define SIMD_STORE2(x, d, h) ((h) ? store_8_f32_as_bf16_nearest(d, x) : _mm256_storeu_ps(x, d))
#else
#define SIMD_LOAD2(x, h) \
    ((h) ? _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(x))) : _mm256_loadu_ps(x))

#define SIMD_STORE2(x, d, h)                                                                \
    ((h) ? _mm_store_ps(x, _mm_castsi128_ps(_mm256_cvtps_ph(d, _MM_FROUND_TO_NEAREST_INT))) \
         : _mm256_storeu_ps(x, d))
#endif
#define INTV __m128i
#endif

union AVX_Data {
#if defined(__AVX512__)
    __m512 data;
#elif defined(__AVX256__)
    __m256 data;
#endif
    // float data_f[16];
};

template <int span>
inline void simd_store(float* dst, AVX_Data* src, bool half_precision)
{
    size_t width = (half_precision ? SIMD_WIDTH / 2 : SIMD_WIDTH);
#pragma unroll
    for (size_t i = 0; i < span; ++i) { SIMD_STORE2(dst + width * i, src[i].data, half_precision); }
}
template <int span>
inline void simd_load(AVX_Data* dst, float* src, bool half_precision)
{
    size_t width = (half_precision ? SIMD_WIDTH / 2 : SIMD_WIDTH);
#pragma unroll
    for (size_t i = 0; i < span; ++i) { dst[i].data = SIMD_LOAD2(src + width * i, half_precision); }
}
template <int span>
inline void simd_fma(AVX_Data* dst, AVX_Data* src_m_l, AVX_Data src_m_r, AVX_Data* src_a)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) {
        dst[i].data = SIMD_FMA(src_m_l[i].data, src_m_r.data, src_a[i].data);
    }
}
template <int span>
inline void simd_fma(AVX_Data* dst, AVX_Data* src_m_l, AVX_Data src_m_r, AVX_Data src_a)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) {
        dst[i].data = SIMD_FMA(src_m_l[i].data, src_m_r.data, src_a.data);
    }
}
template <int span>
inline void simd_fma(AVX_Data* dst, AVX_Data* src_m_l, AVX_Data* src_m_r, AVX_Data* src_a)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) {
        dst[i].data = SIMD_FMA(src_m_l[i].data, src_m_r[i].data, src_a[i].data);
    }
}
template <int span>
inline void simd_sqrt(AVX_Data* dst, AVX_Data* src)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) { dst[i].data = SIMD_SQRT(src[i].data); }
}
template <int span>
inline void simd_add(AVX_Data* dst, AVX_Data* src_a_l, AVX_Data src_a_r)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) { dst[i].data = SIMD_ADD(src_a_l[i].data, src_a_r.data); }
}
template <int span>
inline void simd_add(AVX_Data* dst, AVX_Data* src_a_l, AVX_Data* src_a_r)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) { dst[i].data = SIMD_ADD(src_a_l[i].data, src_a_r[i].data); }
}
template <int span>
inline void simd_mul(AVX_Data* dst, AVX_Data* src_a_l, AVX_Data src_a_r)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) { dst[i].data = SIMD_MUL(src_a_l[i].data, src_a_r.data); }
}
template <int span>
inline void simd_mul(AVX_Data* dst, AVX_Data* src_a_l, AVX_Data* src_a_r)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) { dst[i].data = SIMD_MUL(src_a_l[i].data, src_a_r[i].data); }
}
template <int span>
inline void simd_div(AVX_Data* dst, AVX_Data* src_a_l, AVX_Data* src_a_r)
{
#pragma unroll
    for (size_t i = 0; i < span; ++i) { dst[i].data = SIMD_DIV(src_a_l[i].data, src_a_r[i].data); }
}

#endif
