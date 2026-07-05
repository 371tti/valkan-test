#ifndef REBUILD1_POST_BLOOM_COMMON_GLSL
#define REBUILD1_POST_BLOOM_COMMON_GLSL

#include "common_math.glsl"

float bloom_highlight_weight(vec3 color, float threshold) {
    color = max(color, vec3(0.0));
    float luma = luminance_of(color);
    float knee = max(threshold * 0.45, 0.001);
    float soft = clamp(luma - threshold + knee, 0.0, knee * 2.0);
    float contribution = max(luma - threshold, soft * soft * rcp_safe(knee * 4.0, 0.0001));

    return saturate(contribution * rcp_safe(luma, 0.0001));
}

vec3 bloom_extract(vec3 color, float threshold) {
    return max(color, vec3(0.0)) * bloom_highlight_weight(color, threshold);
}

#endif
