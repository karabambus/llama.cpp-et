//******************************************************************************
// ET Floating Point Math Library
// Provides ET hardware-specific math functions, FP16 conversion, and trig functions
// for bare metal kernels
//******************************************************************************

#ifndef MATH_FP_H
#define MATH_FP_H

#include <stdint.h>

//******************************************************************************
// ET Hardware Math Functions
//******************************************************************************

// ET hardware division function (uses FRCP.PS instruction)
static inline float et_fdiv(float a, float b) {
    float         d;
    unsigned long temp;

    __asm__ volatile(
        "mova.x.m  %[temp]              \n\t"
        "mov.m.x   m0, x0, 1            \n\t"
        "frcp.ps   %[d], %[b]           \n\t"
        "fmul.s    %[d], %[d], %[a]     \n\t"
        "mova.m.x  %[temp]              \n\t"
        : [temp] "=&r"(temp), [d] "=&f"(d)
        : [a] "f"(a), [b] "f"(b));

    return d;
}

// Power function using ET hardware vector instructions
// Implements pow(base, exp) = exp(exp * ln(base)) using FLOG.PS and FEXP.PS
static inline float et_powf(float base, float exp) {
    // Handle special cases
    if (base <= 0.0f) {
        if (base == 0.0f) {
            if (exp > 0.0f) {
                return 0.0f;
            }

            // For exp <= 0, return +infinity (IEEE 754: sign=0, exp=0xFF, mantissa=0)
            union {
                float    f;
                uint32_t i;
            } inf = { .i = 0x7F800000 };

            return inf.f;
        }

        // For negative base, return NaN (IEEE 754: exp=0xFF, mantissa!=0)
        union {
            float    f;
            uint32_t i;
        } nan = { .i = 0x7FC00000 };

        return nan.f;
    }
    if (base == 1.0f) {
        return 1.0f;
    }
    if (exp == 0.0f) {
        return 1.0f;
    }
    if (exp == 1.0f) {
        return base;
    }

    // Use ET hardware instructions following DNN library pattern:
    // pow(base, exp) = exp(exp * ln(base))
    float         result;
    unsigned long temp;

    __asm__ volatile(
        "mova.x.m  %[temp]              \n\t"      // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"      // Set mask register m0 to enable element 0
        "flog.ps %[result], %[base]     \n\t"      // result = ln(base)
        "fmul.s %[result], %[result], %[exp]\n\t"  // result = ln(base) * exp
        "fexp.ps %[result], %[result]   \n\t"      // result = exp(ln(base) * exp) = base^exp
        "mova.m.x  %[temp]              \n\t"      // Restore mask state
        : [temp] "=&r"(temp), [result] "=&f"(result)
        : [base] "f"(base), [exp] "f"(exp));

    return result;
}

// Natural logarithm.
static inline float et_logf(float x) {
    // Handle special cases
    if (x < 0.0f) {
        // Return NaN for negative input
        union {
            float    f;
            uint32_t i;
        } nan = { .i = 0x7FC00000 };

        return nan.f;
    }
    if (x == 0.0f) {
        // Return -infinity for log(0)
        union {
            float    f;
            uint32_t i;
        } inf = { .i = 0xFF800000 };

        return inf.f;
    }
    if (x == 1.0f) {
        return 0.0f;
    }

    float         log2_result;
    unsigned long temp;

    __asm__ volatile(
        "mova.x.m  %[temp]              \n\t"  // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"  // Set mask register m0 to enable element 0
        "flog.ps %[result], %[x]        \n\t"  // result = log2(x)
        "mova.m.x  %[temp]              \n\t"  // Restore mask state
        : [temp] "=&r"(temp), [result] "=&f"(log2_result)
        : [x] "f"(x));

    // Convert log2 to natural log: ln(x) = log2(x) * ln(2)
    const float ln2 = 0.69314718055994530942f;
    return log2_result * ln2;
}

// Square root function implemented as et_powf(x, 0.5)
static inline float et_sqrtf(float x) {
    // Handle special cases
    if (x < 0.0f) {
        // Return NaN for negative input (IEEE 754: exp=0xFF, mantissa!=0)
        union {
            float    f;
            uint32_t i;
        } nan = { .i = 0x7FC00000 };

        return nan.f;
    }
    if (x == 0.0f) {
        return 0.0f;
    }

    return et_powf(x, 0.5f);
}

// Base-2 exponential: returns 2^x using the ET hardware FEXP.PS instruction.
// No base conversion, no special-case clamping — this is the raw hardware op
// with just the mask save/restore wrapper.  Caller is responsible for ensuring
// x is in a range that produces a useful result (roughly [-126, 128] for fp32).
static inline float __attribute__((always_inline)) et_exp2f(float x) {
    unsigned long old_mask;
    float         out;
    __asm__ volatile(
        "mova.x.m  %[ms]             \n\t"
        "mov.m.x   m0, x0, 1         \n\t"
        "fexp.ps   %[out], %[x]      \n\t"
        "mova.m.x  %[ms]             \n\t"
        : [ms] "=&r"(old_mask), [out] "=&f"(out)
        : [x] "f"(x));
    return out;
}

// Exponential function using ET hardware FEXP.PS instruction
// Note: FEXP.PS computes 2^x, so we need to convert: exp(x) = 2^(x * log2(e))
static inline float et_expf(float x) {
    // Handle special cases
    if (x > 88.0f) {
        // For x > 88, exp(x) would overflow, return +infinity
        union {
            float    f;
            uint32_t i;
        } inf = { .i = 0x7F800000 };

        return inf.f;
    }
    if (x < -87.0f) {
        // For x < -87, exp(x) is essentially 0
        return 0.0f;
    }

    // Convert to base-2 exponent: x * log2(e)
    const float log2e   = 1.4426950408889634f;  // log2(e)
    float       x_log2e = x * log2e;

    // Use ET hardware instruction: fexp.ps computes 2^x
    float         result;
    unsigned long temp;

    __asm__ volatile(
        "mova.x.m  %[temp]              \n\t"  // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"  // Set mask register m0 to enable element 0
        "fexp.ps %[result], %[x_log2e]  \n\t"  // result = 2^(x * log2(e)) = exp(x)
        "mova.m.x  %[temp]              \n\t"  // Restore mask state
        : [temp] "=&r"(temp), [result] "=&f"(result)
        : [x_log2e] "f"(x_log2e));

    return result;
}

//******************************************************************************
// Trigonometric Functions
//******************************************************************************

// FSIN.PS

// Sine function using Taylor series
static inline float et_sinf(float x) {
    const float pi        = 3.14159265358979323846f;
    const float two_pi    = 6.28318530717958647693f;
    const float pi_over_2 = 1.57079632679489661923f;

    if (x > pi || x < -pi) {
        float cycles = x * et_fdiv(1.0f, two_pi);
        int   n      = (int) cycles;
        if (x < 0.0f) {
            n--;  // Floor for negative
        }
        x = x - (float) n * two_pi;
    }

    // sin(x) = sin(π - x) for x in [π/2, π]
    // sin(x) = -sin(-π - x) for x in [-π, -π/2]
    int negate = 0;
    if (x > pi_over_2) {
        x = pi - x;
    } else if (x < -pi_over_2) {
        x      = -pi - x;
        negate = 1;
    }

    // sin(x) ≈ x - x^3/3! + x^5/5! - x^7/7! + x^9/9! - x^11/11!
    const float x2  = x * x;
    const float x3  = x2 * x;
    const float x5  = x3 * x2;
    const float x7  = x5 * x2;
    const float x9  = x7 * x2;
    const float x11 = x9 * x2;

    float result = x - x3 * et_fdiv(1.0f, 6.0f)         // x^3/3!
                   + x5 * et_fdiv(1.0f, 120.0f)         // x^5/5!
                   - x7 * et_fdiv(1.0f, 5040.0f)        // x^7/7!
                   + x9 * et_fdiv(1.0f, 362880.0f)      // x^9/9!
                   - x11 * et_fdiv(1.0f, 39916800.0f);  // x^11/11!

    return negate ? -result : result;
}

// Cosine function using identity cos(x) = sin(x + π/2)
static inline float et_cosf(float x) {
    const float pi_over_2 = 1.57079632679489661923f;
    return et_sinf(x + pi_over_2);
}

//******************************************************************************
// FP16 <-> FP32 Conversion Functions
//******************************************************************************

// Convert FP16 (IEEE 754 half precision) to FP32 (single precision), exactly, in integer code.
//
// Not FCVT.PS.F16: that instruction flushes subnormal halves (|x| < 2^-14) to zero, and quantised block
// scales reach that range. A Q8_0 scale is max|w| / 127, so blocks of small weights carry subnormal scales
// (Qwen3.6-35B-A3B shared-expert tensors: 0.5 % of blocks, up to 3.5 % in one tensor); flushing them zeroes
// whole weight blocks. Callers in vector code broadcast fp16_to_fp32_bits() with FBCX.PS instead of converting
// in the lanes. Normal, zero, infinity and NaN inputs convert bit-identically to the hardware instruction.
static inline uint32_t fp16_to_fp32_bits_slow(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    uint32_t       exp  = (h >> 10) & 0x1Fu;
    uint32_t       mant = h & 0x3FFu;
    if (exp == 0x1Fu) {                              // inf / nan
        return sign | 0x7F800000u | (mant << 13);
    }
    if (mant == 0) {                                 // +-0
        return sign;
    }
    exp = 127 - 15 + 1;                              // subnormal: normalise
    while ((mant & 0x400u) == 0) {
        mant <<= 1;
        exp--;
    }
    mant &= 0x3FFu;
    return sign | (exp << 23) | (mant << 13);
}

// Normal halves (exponent field 1..30) are a shift and an exponent rebias: ((h & 0x7fff) + (112 << 10)) << 13 moves
// sign-less exponent+mantissa into place and adds 127 - 15 to the exponent; the sign bit moves separately. Only
// zeros, subnormals, infinities and NaNs take the branchy path. This is the hot path of the matrix-engine dequant
// panels (one scale per weight row per 32-wide block).
static inline uint32_t __attribute__((always_inline)) fp16_to_fp32_bits(uint16_t h) {
    const uint32_t e = h & 0x7C00u;
    if (__builtin_expect(e != 0 && e != 0x7C00u, 1)) {
        return ((uint32_t) (h & 0x8000u) << 16) | (((uint32_t) (h & 0x7FFFu) + (112u << 10)) << 13);
    }
    return fp16_to_fp32_bits_slow(h);
}

static inline float fp16_to_fp32(uint16_t h) {
    const uint32_t bits = fp16_to_fp32_bits(h);
    float          out;
    __builtin_memcpy(&out, &bits, 4);
    return out;
}

// Convert FP32 (single precision) to FP16 (IEEE 754 half precision)
// Uses ET hardware FCVT.F16.PS instruction for accurate conversion
static inline uint16_t fp32_to_fp16(float f) {
    float         result_f;
    unsigned long temp;

    __asm__ volatile(
        "mova.x.m  %[temp]              \n\t"  // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"  // Set mask register m0 to enable element 0
        "fcvt.f16.ps %[result], %[f]    \n\t"  // Convert FP32 to FP16 (result in lower 16 bits)
        "mova.m.x  %[temp]              \n\t"  // Restore mask state
        : [temp] "=&r"(temp), [result] "=&f"(result_f)
        : [f] "f"(f));

    // Extract lower 16 bits containing the FP16 value
    // The instruction zero-extends to 32 bits, so upper 16 bits are 0
    uint32_t result_bits = *(uint32_t *) &result_f;
    return (uint16_t) result_bits;
}

#endif  // MATH_FP_H
