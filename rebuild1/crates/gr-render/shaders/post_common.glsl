#ifndef REBUILD1_POST_COMMON_GLSL
#define REBUILD1_POST_COMMON_GLSL

float saturate(float value) {
    return clamp(value, 0.0, 1.0);
}

vec3 saturate(vec3 value) {
    return clamp(value, vec3(0.0), vec3(1.0));
}

float rcp_safe(float value, float floor_value) {
    return 1.0 / max(value, floor_value);
}

vec3 normalize_fast(vec3 value) {
    return value * inversesqrt(max(dot(value, value), 0.000001));
}

vec3 filmic_tonemap(vec3 color) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;

    vec3 numerator = color * (a * color + b);
    vec3 denominator = color * (c * color + d) + e;

    return saturate(numerator / max(denominator, vec3(0.0001)));
}

vec3 apply_camera_effects(vec3 color) {
    if (params.enabled < 0.5) {
        return max(color, vec3(0.0));
    }

    color = max(color, vec3(0.0)) * params.exposure * params.white_balance.rgb;
    color = filmic_tonemap(color);

    float luminance = dot(color, vec3(0.2126, 0.7152, 0.0722));
    color = mix(vec3(luminance), color, params.saturation);

    const vec3 contrast_pivot = vec3(0.36);
    color = contrast_pivot + (color - contrast_pivot) * params.contrast;

    return saturate(color);
}

float linear_depth(float depth) {
    return params.depth.z * rcp_safe(params.depth.y - depth * params.depth.w, 0.0001);
}

vec3 view_position(vec2 uv, float depth) {
    float z = linear_depth(depth);
    vec2 ndc = uv * 2.0 - 1.0;

    return vec3(
        ndc.x * z * params.camera.z,
        ndc.y * z * params.camera.w,
        -z
    );
}

vec2 project_view_position(vec3 position) {
    float inv_depth = rcp_safe(-position.z, 0.0001);
    vec2 ndc = vec2(
        position.x * params.camera.x * inv_depth,
        position.y * params.camera.y * inv_depth
    );

    return ndc * 0.5 + 0.5;
}

float screen_edge_fade(vec2 uv) {
    vec2 edge = min(uv, vec2(1.0) - uv);
    return smoothstep(0.02, 0.18, min(edge.x, edge.y));
}

float depth_at(vec2 uv) {
    return texture(scene_depth, uv).r;
}

float depth_at_ssr(vec2 uv) {
    vec2 texel = params.aa.xy;
    float center = depth_at(uv);
    float left = depth_at(uv + vec2(-texel.x, 0.0));
    float right = depth_at(uv + vec2(texel.x, 0.0));
    float up = depth_at(uv + vec2(0.0, -texel.y));
    float down = depth_at(uv + vec2(0.0, texel.y));

    float min_depth = min(center, min(min(left, right), min(up, down)));
    float max_depth = max(center, max(max(left, right), max(up, down)));
    float stable = (center * 2.0 + left + right + up + down) * 0.16666667;
    float edge = smoothstep(0.0004, 0.0045, max_depth - min_depth);

    return mix(stable, center, edge);
}

float luminance_of(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

float perceptual_luma(vec3 color) {
    float luma = luminance_of(max(color, vec3(0.0)));
    return luma * rcp_safe(1.0 + luma, 0.0001);
}

vec2 sign_not_zero(vec2 value) {
    return vec2(
        value.x >= 0.0 ? 1.0 : -1.0,
        value.y >= 0.0 ? 1.0 : -1.0
    );
}

vec3 oct_decode(vec2 encoded) {
    vec2 f = encoded * 2.0 - 1.0;
    vec3 normal = vec3(f, 1.0 - abs(f.x) - abs(f.y));

    if (normal.z < 0.0) {
        normal.xy = (1.0 - abs(normal.yx)) * sign_not_zero(normal.xy);
    }

    return normalize_fast(normal);
}

struct SurfaceMaterial {
    vec3 normal;
    float source_depth;
    float roughness;
    float reflectance;
    float occlusion_weight;
};

SurfaceMaterial empty_surface_material(float depth) {
    return SurfaceMaterial(vec3(0.0, 0.0, 1.0), depth, 1.0, 0.0, 0.0);
}

SurfaceMaterial opaque_surface_material(vec4 packed, float depth) {
    return SurfaceMaterial(
        oct_decode(packed.xy),
        depth,
        clamp(packed.z, 0.04, 1.0),
        clamp(packed.w, 0.0, 1.0),
        1.0
    );
}

SurfaceMaterial transparent_surface_material(vec4 packed) {
    float roughness = clamp(packed.z, 0.04, 1.0);
    return SurfaceMaterial(
        oct_decode(packed.xy),
        clamp(packed.w - 1.0, 0.0, 0.9999),
        roughness,
        max(0.08, (1.0 - roughness) * 0.28),
        0.25
    );
}

SurfaceMaterial surface_material(vec2 uv) {
    float opaque_depth = depth_at(uv);
    vec4 transparent = texture(scene_transparent_normal_roughness, uv);
    bool has_transparent = transparent.w > 1.0;

    if (opaque_depth >= 0.9999) {
        if (has_transparent) {
            return transparent_surface_material(transparent);
        }
        return empty_surface_material(opaque_depth);
    }

    vec4 opaque = texture(scene_normal_roughness, uv);

    if (has_transparent) {
        float transparent_depth = clamp(transparent.w - 1.0, 0.0, 0.9999);
        if (transparent_depth <= opaque_depth + 0.00001) {
            return transparent_surface_material(transparent);
        }
    }

    return opaque_surface_material(opaque, opaque_depth);
}

const float SHADOW_BLUR_KERNEL_5[5] = float[5](
    0.06136,
    0.24477,
    0.38774,
    0.24477,
    0.06136
);

float shadow_soften_weight(
    float kernel_weight,
    SurfaceMaterial center,
    SurfaceMaterial sample_material,
    float center_view_depth,
    float sample_view_depth,
    float depth_reject_scale
) {
    float depth_delta = abs(sample_view_depth - center_view_depth);
    float depth_weight = exp2(-depth_delta * depth_reject_scale);

    float normal_match = dot(center.normal, sample_material.normal);
    float normal_weight = smoothstep(0.72, 0.98, normal_match);

    return kernel_weight * depth_weight * normal_weight;
}

vec3 shadow_softened_scene_color(vec2 uv, vec3 color, SurfaceMaterial material) {
    if (params.shadow.x <= 0.0 || material.source_depth >= 0.9999) {
        return color;
    }

    float center_luma = perceptual_luma(color);
    float center_view_depth = linear_depth(material.source_depth);
    float depth_scale = max(center_view_depth * 0.010, 0.015);
    float depth_reject_scale = params.shadow.z * rcp_safe(depth_scale, 0.0001);
    float radius = max(params.shadow.y, 0.5);
    vec2 texel_radius = params.aa.xy * radius;
    int half_width = radius < 1.5 ? 1 : 2;

    float luma_sum = center_luma * SHADOW_BLUR_KERNEL_5[2] * SHADOW_BLUR_KERNEL_5[2];
    float weight_sum = SHADOW_BLUR_KERNEL_5[2] * SHADOW_BLUR_KERNEL_5[2];

    for (int y = -2; y <= 2; y++) {
        for (int x = -2; x <= 2; x++) {
            if (x == 0 && y == 0) {
                continue;
            }

            if (abs(x) > half_width || abs(y) > half_width) {
                continue;
            }

            vec2 offset = vec2(float(x), float(y));
            float kernel_weight =
                SHADOW_BLUR_KERNEL_5[x + 2] *
                SHADOW_BLUR_KERNEL_5[y + 2];
            vec2 sample_uv = clamp(uv + offset * texel_radius, vec2(0.0), vec2(1.0));
            SurfaceMaterial sample_material = surface_material(sample_uv);

            if (sample_material.source_depth >= 0.9999) {
                continue;
            }

            vec3 sample_color = texture(scene_color, sample_uv).rgb;
            float sample_luma = perceptual_luma(sample_color);
            float sample_view_depth = linear_depth(sample_material.source_depth);
            float weight = shadow_soften_weight(
                kernel_weight,
                material,
                sample_material,
                center_view_depth,
                sample_view_depth,
                depth_reject_scale
            );

            luma_sum += sample_luma * weight;
            weight_sum += weight;
        }
    }

    float filtered_luma = luma_sum * rcp_safe(weight_sum, 0.0001);
    float max_luma_delta = max(params.shadow.w, 0.005);
    float luma_delta = clamp(
        filtered_luma - center_luma,
        -max_luma_delta,
        max_luma_delta
    );
    float luma_signal = smoothstep(0.001, max_luma_delta * 0.35, abs(luma_delta));
    float target_luma = max(center_luma + luma_delta, 0.0);
    float luma_ratio = target_luma * rcp_safe(center_luma, 0.0001);
    vec3 adjusted = color * clamp(luma_ratio, 0.45, 1.85);
    float blend = clamp(params.shadow.x, 0.0, 1.0) * luma_signal;

    return mix(color, adjusted, blend);
}

#endif
