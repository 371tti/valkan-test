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

#endif
