#pragma once

float3 filmic_tonemap(float3 color) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;

    color = max(color, float3(0.0));

    float3 numerator = color * (a * color + b);
    float3 denominator = color * (c * color + d) + e;

    return saturate(numerator / max(denominator, float3(0.0001)));
}

float3 apply_camera_effects(float3 color) {
    if (params.enabled < 0.5) {
        return max(color, float3(0.0));
    }

    color = max(color, float3(0.0));
    color *= params.exposure;
    color *= params.white_balance.rgb;

    color = filmic_tonemap(color);

    float luminance = dot(color, float3(0.2126, 0.7152, 0.0722));
    color = lerp(float3(luminance), color, params.saturation);

    const float3 contrast_pivot = float3(0.36);
    color = contrast_pivot + (color - contrast_pivot) * params.contrast;

    return saturate(color);
}
