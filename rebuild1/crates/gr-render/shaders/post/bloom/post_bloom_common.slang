#pragma once

#include "common_math.slang"

float bloom_highlight_weight(float3 color, float threshold) {
    color = max(color, float3(0.0));
    float luma = luminance_of(color);
    float knee = max(threshold * 0.45, 0.001);
    float soft = clamp(luma - threshold + knee, 0.0, knee * 2.0);
    float contribution = max(luma - threshold, soft * soft * rcp_safe(knee * 4.0, 0.0001));

    return saturate(contribution * rcp_safe(luma, 0.0001));
}

float3 bloom_extract(float3 color, float threshold) {
    return max(color, float3(0.0)) * bloom_highlight_weight(color, threshold);
}

