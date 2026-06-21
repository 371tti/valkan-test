#ifndef REBUILD1_POST_COMMON_GLSL
#define REBUILD1_POST_COMMON_GLSL

#include "common_math.glsl"

const float POST_BACKGROUND_DEPTH = 0.9999;

vec2 clamp_screen_uv(vec2 uv) {
    return clamp(uv, vec2(0.0), vec2(1.0));
}

vec3 sample_scene_color(vec2 uv) {
    return texture(scene_color, clamp_screen_uv(uv)).rgb;
}

float depth_at(vec2 uv) {
    return texture(scene_depth, clamp_screen_uv(uv)).r;
}

bool is_background_depth(float depth) {
    return depth >= POST_BACKGROUND_DEPTH;
}

vec3 filmic_tonemap(vec3 color) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;

    color = max(color, vec3(0.0));

    vec3 numerator = color * (a * color + b);
    vec3 denominator = color * (c * color + d) + e;

    return saturate(numerator / max(denominator, vec3(0.0001)));
}

vec3 apply_camera_effects(vec3 color) {
    if (params.enabled < 0.5) {
        return max(color, vec3(0.0));
    }

    color = max(color, vec3(0.0));
    color *= params.exposure;
    color *= params.white_balance.rgb;

    color = filmic_tonemap(color);

    float luminance = dot(color, vec3(0.2126, 0.7152, 0.0722));
    color = mix(vec3(luminance), color, params.saturation);

    const vec3 contrast_pivot = vec3(0.36);
    color = contrast_pivot + (color - contrast_pivot) * params.contrast;

    return saturate(color);
}

float linear_depth(float depth) {
    return params.depth.z * rcp_safe(
        params.depth.y - depth * params.depth.w,
        0.0001
    );
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

bool post_ssao_enabled() {
    return params.ssao.x > 0.0;
}

bool post_ssr_enabled() {
    return params.ssr.x > 0.0 && params.ssr.y >= 1.0;
}

bool post_aa_enabled() {
    return params.aa.w > 0.0;
}

bool post_contact_shadow_enabled() {
    return params.shadow.x > 0.0 &&
        params.shadow.y > 0.0 &&
        params.shadow.w >= 1.0;
}

bool post_transparent_metadata_enabled() {
    return params.features.x > 0.5;
}

bool post_requires_surface_material() {
    return post_ssao_enabled() ||
        post_ssr_enabled() ||
        post_contact_shadow_enabled();
}

float depth_at_ssr(vec2 uv) {
    vec2 texel = params.aa.xy;

    float center = depth_at(uv);

    if (is_background_depth(center)) {
        return center;
    }

    float left = depth_at(uv + vec2(-texel.x, 0.0));
    float right = depth_at(uv + vec2(texel.x, 0.0));
    float up = depth_at(uv + vec2(0.0, -texel.y));
    float down = depth_at(uv + vec2(0.0, texel.y));

    float min_depth = min(center, min(min(left, right), min(up, down)));
    float max_depth = max(center, max(max(left, right), max(up, down)));

    float stable = (
        center * 2.0 +
        left +
        right +
        up +
        down
    ) * 0.16666667;

    float depth_range = max_depth - min_depth;

    float edge = smoothstep(
        0.0004,
        0.0045,
        depth_range
    );

    return mix(stable, center, edge);
}

float depth_at_ssr_trace(vec2 uv) {
    return depth_at(uv);
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
    float transparent_weight;
};

SurfaceMaterial empty_surface_material(float depth) {
    return SurfaceMaterial(
        vec3(0.0, 0.0, 1.0),
        depth,
        1.0,
        0.0,
        0.0,
        0.0
    );
}

bool surface_material_is_background(SurfaceMaterial material) {
    return is_background_depth(material.source_depth);
}

SurfaceMaterial opaque_surface_material(vec4 packed, float depth) {
    return SurfaceMaterial(
        oct_decode(packed.xy),
        depth,
        clamp(packed.z, 0.04, 1.0),
        clamp(packed.w, 0.0, 1.0),
        1.0,
        0.0
    );
}

SurfaceMaterial transparent_surface_material(vec4 packed) {
    float roughness = clamp(packed.z, 0.04, 1.0);

    return SurfaceMaterial(
        oct_decode(packed.xy),
        clamp(packed.w - 1.0, 0.0, POST_BACKGROUND_DEPTH),
        roughness,
        max(0.08, (1.0 - roughness) * 0.28),
        0.25,
        1.0
    );
}

SurfaceMaterial surface_material(vec2 uv) {
    uv = clamp_screen_uv(uv);

    float opaque_depth = depth_at(uv);

    if (!post_transparent_metadata_enabled()) {
        if (is_background_depth(opaque_depth)) {
            return empty_surface_material(opaque_depth);
        }

        return opaque_surface_material(
            texture(scene_normal_roughness, uv),
            opaque_depth
        );
    }

    vec4 transparent = texture(scene_transparent_normal_roughness, uv);
    bool has_transparent = transparent.w > 1.0;

    if (is_background_depth(opaque_depth)) {
        if (has_transparent) {
            return transparent_surface_material(transparent);
        }

        return empty_surface_material(opaque_depth);
    }

    vec4 opaque = texture(scene_normal_roughness, uv);

    if (has_transparent) {
        float transparent_depth = clamp(
            transparent.w - 1.0,
            0.0,
            POST_BACKGROUND_DEPTH
        );

        if (transparent_depth <= opaque_depth + 0.00001) {
            return transparent_surface_material(transparent);
        }
    }

    return opaque_surface_material(opaque, opaque_depth);
}

SurfaceMaterial load_feature_surface_material(vec2 uv, out bool valid) {
    valid = post_requires_surface_material();

    return valid
        ? surface_material(uv)
        : empty_surface_material(POST_BACKGROUND_DEPTH);
}

// =========================================================
// Screen-space contact shadows
// =========================================================

const int CONTACT_SHADOW_MAX_STEPS = 24;

float contact_shadow_sample_occlusion(
    vec3 ray_position,
    vec2 sample_uv,
    float ray_distance,
    float base_thickness,
    vec3 receiver_normal
) {
    if (
        sample_uv.x <= 0.0 || sample_uv.x >= 1.0 ||
        sample_uv.y <= 0.0 || sample_uv.y >= 1.0 ||
        ray_position.z >= -params.depth.x
    ) {
        return 0.0;
    }

    float sample_depth = depth_at(sample_uv);

    if (is_background_depth(sample_depth)) {
        return 0.0;
    }

    float ray_view_depth = -ray_position.z;
    float scene_view_depth = linear_depth(sample_depth);
    float depth_delta = ray_view_depth - scene_view_depth;
    float self_bias =
        base_thickness * 0.55 +
        ray_distance * 0.010 +
        scene_view_depth * 0.00018;
    float thickness =
        base_thickness +
        ray_distance * 0.090 +
        scene_view_depth * 0.00055;

    if (depth_delta <= self_bias || depth_delta >= thickness * 1.35) {
        return 0.0;
    }

    float front = smoothstep(self_bias, self_bias + base_thickness * 0.70, depth_delta);
    float back = 1.0 - smoothstep(thickness * 0.72, thickness * 1.35, depth_delta);
    float occlusion = front * back;

    vec3 sample_normal = oct_decode(texture(scene_normal_roughness, clamp_screen_uv(sample_uv)).xy);
    float same_normal = smoothstep(0.90, 0.985, dot(sample_normal, receiver_normal));
    float shallow_hit = 1.0 - smoothstep(self_bias + base_thickness, thickness, depth_delta);

    return occlusion * (1.0 - same_normal * shallow_hit * 0.88);
}

float screen_space_contact_shadow(vec2 uv, SurfaceMaterial material) {
    if (
        !post_contact_shadow_enabled() ||
        surface_material_is_background(material) ||
        material.transparent_weight > 0.75
    ) {
        return 1.0;
    }

    vec3 origin = view_position(uv, material.source_depth);
    float view_depth = -origin.z;

    if (view_depth <= params.depth.x) {
        return 1.0;
    }

    vec3 normal = normalize_fast(material.normal);
    vec3 light_dir = normalize_fast(params.features.yzw);
    float ndotl = saturate(dot(normal, light_dir));

    if (ndotl <= 0.025) {
        return 1.0;
    }

    int sample_count = int(clamp(params.shadow.w, 1.0, float(CONTACT_SHADOW_MAX_STEPS)));
    float max_distance = max(params.shadow.y, 0.05);
    float base_thickness = max(params.shadow.z, 0.008);
    vec3 ray_origin =
        origin +
        normal * max(base_thickness * 0.48, view_depth * 0.00070) +
        light_dir * max(base_thickness * 0.60, 0.014);
    float transmittance = 1.0;

    for (int i = 0; i < CONTACT_SHADOW_MAX_STEPS; i++) {
        if (i >= sample_count) {
            break;
        }

        float step_t = (float(i) + 0.5) / float(sample_count);
        float ray_distance = max_distance * pow(step_t, 1.35);
        vec3 ray_position = ray_origin + light_dir * ray_distance;
        vec2 sample_uv = project_view_position(ray_position);
        float sample_occlusion = contact_shadow_sample_occlusion(
            ray_position,
            sample_uv,
            ray_distance,
            base_thickness,
            normal
        );
        float distance_fade =
            1.0 - smoothstep(max_distance * 0.34, max_distance, ray_distance);
        float step_weight = mix(0.68, 0.18, step_t);

        transmittance *= 1.0 - saturate(sample_occlusion * distance_fade * step_weight);
    }

    float light_fade = smoothstep(0.06, 0.42, ndotl);
    float depth_fade = 1.0 - smoothstep(48.0, 140.0, view_depth);
    float strength =
        clamp(params.shadow.x, 0.0, 1.0) *
        material.occlusion_weight *
        light_fade *
        depth_fade;

    float occlusion = 1.0 - transmittance;

    return 1.0 - saturate(occlusion * strength * 1.45);
}

vec3 contact_shadowed_scene_color(vec2 uv, vec3 color, SurfaceMaterial material) {
    float visibility = screen_space_contact_shadow(uv, material);
    return color * visibility;
}
#endif
