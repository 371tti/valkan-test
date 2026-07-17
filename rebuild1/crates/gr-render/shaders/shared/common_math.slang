#pragma once

#include "slang_compat.slang"

float saturate(float value) {
    return clamp(value, 0.0, 1.0);
}

float3 saturate(float3 value) {
    return clamp(value, float3(0.0), float3(1.0));
}

float pow5(float value) {
    float value2 = value * value;
    return value2 * value2 * value;
}

float rcp_safe(float value, float floor_value) {
    return 1.0 / max(value, floor_value);
}

float3 normalize_fast(float3 value) {
    return value * rsqrt(max(dot(value, value), 0.000001));
}

float2 normalize2_fast(float2 value) {
    return value * rsqrt(max(dot(value, value), 0.000001));
}

float2 sign_not_zero(float2 value) {
    return float2(
        value.x >= 0.0 ? 1.0 : -1.0,
        value.y >= 0.0 ? 1.0 : -1.0
    );
}

float luminance_of(float3 color) {
    return dot(color, float3(0.2126, 0.7152, 0.0722));
}

float perceptual_luma(float3 color) {
    float luma = luminance_of(max(color, float3(0.0)));
    return luma * rcp_safe(1.0 + luma, 0.0001);
}
