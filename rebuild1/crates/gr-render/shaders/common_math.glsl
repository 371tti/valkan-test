#ifndef REBUILD1_COMMON_MATH_GLSL
#define REBUILD1_COMMON_MATH_GLSL

float saturate(float value) {
    return clamp(value, 0.0, 1.0);
}

vec3 saturate(vec3 value) {
    return clamp(value, vec3(0.0), vec3(1.0));
}

float pow5(float value) {
    float value2 = value * value;
    return value2 * value2 * value;
}

float rcp_safe(float value, float floor_value) {
    return 1.0 / max(value, floor_value);
}

vec3 normalize_fast(vec3 value) {
    return value * inversesqrt(max(dot(value, value), 0.000001));
}

vec2 normalize2_fast(vec2 value) {
    return value * inversesqrt(max(dot(value, value), 0.000001));
}

vec2 sign_not_zero(vec2 value) {
    return vec2(
        value.x >= 0.0 ? 1.0 : -1.0,
        value.y >= 0.0 ? 1.0 : -1.0
    );
}

float luminance_of(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

float perceptual_luma(vec3 color) {
    float luma = luminance_of(max(color, vec3(0.0)));
    return luma * rcp_safe(1.0 + luma, 0.0001);
}

#endif
