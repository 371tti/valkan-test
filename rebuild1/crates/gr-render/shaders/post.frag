#version 450

layout(location = 0) in vec2 frag_uv;
layout(location = 0) out vec4 out_color;

layout(set = 2, binding = 0) uniform sampler2D scene_color;
layout(set = 2, binding = 1) uniform sampler2D scene_depth;
layout(set = 2, binding = 2) uniform sampler2D scene_normal_roughness;

layout(push_constant) uniform PostParams {
    vec4 white_balance;
    vec4 camera;
    vec4 depth;
    vec4 ssao;
    vec4 aa;
    float exposure;
    float contrast;
    float saturation;
    float enabled;
} params;

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
};

SurfaceMaterial surface_material(vec2 uv) {
    float opaque_depth = depth_at(uv);

    if (opaque_depth >= 0.9999) {
        return SurfaceMaterial(vec3(0.0, 0.0, 1.0), opaque_depth);
    }

    vec4 opaque = texture(scene_normal_roughness, uv);
    return SurfaceMaterial(oct_decode(opaque.xy), opaque_depth);
}

float depth_edge_from_depths(float center_depth, float sample_depth) {
    if (center_depth >= 0.9999 && sample_depth >= 0.9999) {
        return 0.0;
    }

    float center_z = linear_depth(center_depth);
    float sample_z = linear_depth(sample_depth);

    float scale = max(min(center_z, sample_z) * 0.018, 0.015);
    return saturate(abs(center_z - sample_z) * rcp_safe(scale, 0.0001));
}

float normal_edge_from_depth(
    vec3 center_normal,
    float center_depth,
    float sample_depth,
    vec2 sample_uv
) {
    // 未描画領域の normal_roughness を oct_decode しない。
    // 空背景や画面端で変な normal edge が出るのを防ぐ。
    if (center_depth >= 0.9999 || sample_depth >= 0.9999) {
        return 0.0;
    }

    vec4 sample_meta = texture(scene_normal_roughness, sample_uv);
    vec3 sample_normal = oct_decode(sample_meta.xy);

    return saturate((1.0 - dot(center_normal, sample_normal)) * 2.4);
}

vec3 high_quality_fxaa_scene_color(
    vec2 uv,
    vec3 center,
    SurfaceMaterial center_material,
    bool center_material_valid
) {
    if (params.aa.w <= 0.0) {
        return center;
    }

    vec2 texel = params.aa.xy;

    vec2 uv_n = uv + vec2(0.0, -texel.y);
    vec2 uv_s = uv + vec2(0.0,  texel.y);
    vec2 uv_w = uv + vec2(-texel.x, 0.0);
    vec2 uv_e = uv + vec2( texel.x, 0.0);

    vec3 north = texture(scene_color, uv_n).rgb;
    vec3 south = texture(scene_color, uv_s).rgb;
    vec3 west  = texture(scene_color, uv_w).rgb;
    vec3 east  = texture(scene_color, uv_e).rgb;

    float luma_m = perceptual_luma(center);
    float luma_n = perceptual_luma(north);
    float luma_s = perceptual_luma(south);
    float luma_w = perceptual_luma(west);
    float luma_e = perceptual_luma(east);

    float luma_min = min(luma_m, min(min(luma_n, luma_s), min(luma_w, luma_e)));
    float luma_max = max(luma_m, max(max(luma_n, luma_s), max(luma_w, luma_e)));

    float edge_signal = luma_max - luma_min;

    float geometry_horizontal = 0.0;
    float geometry_vertical = 0.0;

    if (edge_signal <= params.aa.z) {
        SurfaceMaterial material = center_material;

        if (!center_material_valid) {
            material = surface_material(uv);
        }

        float depth_n = depth_at(uv_n);
        float depth_s = depth_at(uv_s);
        float depth_w = depth_at(uv_w);
        float depth_e = depth_at(uv_e);

        float depth_edge_n = depth_edge_from_depths(material.source_depth, depth_n);
        float depth_edge_s = depth_edge_from_depths(material.source_depth, depth_s);
        float depth_edge_w = depth_edge_from_depths(material.source_depth, depth_w);
        float depth_edge_e = depth_edge_from_depths(material.source_depth, depth_e);

        geometry_horizontal = max(depth_edge_n, depth_edge_s);
        geometry_vertical   = max(depth_edge_w, depth_edge_e);

        float depth_signal = max(
            max(depth_edge_n, depth_edge_s),
            max(depth_edge_w, depth_edge_e)
        );

        float normal_edge_n = normal_edge_from_depth(
            material.normal,
            material.source_depth,
            depth_n,
            uv_n
        );

        float normal_edge_s = normal_edge_from_depth(
            material.normal,
            material.source_depth,
            depth_s,
            uv_s
        );

        float normal_edge_w = normal_edge_from_depth(
            material.normal,
            material.source_depth,
            depth_w,
            uv_w
        );

        float normal_edge_e = normal_edge_from_depth(
            material.normal,
            material.source_depth,
            depth_e,
            uv_e
        );

        float normal_signal = max(
            max(normal_edge_n, normal_edge_s),
            max(normal_edge_w, normal_edge_e)
        );

        edge_signal = max(edge_signal, max(depth_signal, normal_signal) * 0.32);
    }

    if (edge_signal <= params.aa.z) {
        return center;
    }

    vec2 uv_nw = uv + vec2(-texel.x, -texel.y);
    vec2 uv_ne = uv + vec2( texel.x, -texel.y);
    vec2 uv_sw = uv + vec2(-texel.x,  texel.y);
    vec2 uv_se = uv + vec2( texel.x,  texel.y);

    vec3 northwest = texture(scene_color, uv_nw).rgb;
    vec3 northeast = texture(scene_color, uv_ne).rgb;
    vec3 southwest = texture(scene_color, uv_sw).rgb;
    vec3 southeast = texture(scene_color, uv_se).rgb;

    float luma_nw = perceptual_luma(northwest);
    float luma_ne = perceptual_luma(northeast);
    float luma_sw = perceptual_luma(southwest);
    float luma_se = perceptual_luma(southeast);

    luma_min = min(luma_min, min(min(luma_nw, luma_ne), min(luma_sw, luma_se)));
    luma_max = max(luma_max, max(max(luma_nw, luma_ne), max(luma_sw, luma_se)));

    vec2 dir = vec2(
        -((luma_nw + luma_ne) - (luma_sw + luma_se)),
         ((luma_nw + luma_sw) - (luma_ne + luma_se))
    );

    if (dot(dir, dir) <= 0.000001) {
        dir = geometry_vertical > geometry_horizontal
            ? vec2(0.0, 1.0)
            : vec2(1.0, 0.0);
    }

    float reduce = max(
        (luma_nw + luma_ne + luma_sw + luma_se) * 0.0078125,
        0.0009765625
    );

    float scale = rcp_safe(min(abs(dir.x), abs(dir.y)) + reduce, 0.000001);
    dir = clamp(dir * scale, vec2(-8.0), vec2(8.0)) * texel;

    vec3 rgb_a = 0.5 * (
        texture(scene_color, uv + dir * -0.16666667).rgb +
        texture(scene_color, uv + dir *  0.16666667).rgb
    );

    vec3 rgb_b = rgb_a * 0.5 + 0.25 * (
        texture(scene_color, uv + dir * -0.5).rgb +
        texture(scene_color, uv + dir *  0.5).rgb
    );

    float luma_b = perceptual_luma(rgb_b);

    vec3 resolved = (luma_b < luma_min || luma_b > luma_max)
        ? rgb_a
        : rgb_b;

    float blend = saturate(
        (edge_signal - params.aa.z) * rcp_safe(edge_signal, 0.0001)
    ) * params.aa.w;

    return mix(center, resolved, blend);
}

float screen_space_ao(vec2 uv, SurfaceMaterial material) {
    if (material.source_depth >= 0.9999 || params.ssao.x <= 0.0) {
        return 1.0;
    }

    vec3 position = view_position(uv, material.source_depth);
    float view_z = max(-position.z, params.depth.x);

    float radius = max(params.ssao.y, 0.001);
    float radius_sq = radius * radius;
    float bias = max(params.ssao.z, 0.0005);

    float radius_uv = clamp(
        radius * rcp_safe(view_z, 0.0001) * 0.55,
        0.0012,
        0.038
    );

    int sample_count = int(clamp(params.ssao.w, 1.0, 8.0));
    float inv_sample_count = rcp_safe(float(sample_count), 1.0);

    const vec2 kernel[8] = vec2[8](
        vec2( 0.5381,  0.1856),
        vec2(-0.4319,  0.3141),
        vec2( 0.2486, -0.7242),
        vec2(-0.7198, -0.1937),
        vec2( 0.9103,  0.4125),
        vec2(-0.3627, -0.9184),
        vec2( 0.1269,  0.9872),
        vec2(-0.9715,  0.0524)
    );

    float occlusion = 0.0;
    float total = 0.0;

    for (int i = 0; i < 8; i++) {
        if (i >= sample_count) {
            break;
        }

        float scale = mix(
            0.28,
            1.0,
            (float(i) + 1.0) * inv_sample_count
        );

        vec2 sample_uv = uv + kernel[i] * radius_uv * scale;

        if (
            any(lessThan(sample_uv, vec2(0.0))) ||
            any(greaterThan(sample_uv, vec2(1.0)))
        ) {
            continue;
        }

        float sample_depth = depth_at(sample_uv);

        if (sample_depth >= 0.9999) {
            continue;
        }

        vec3 sample_position = view_position(sample_uv, sample_depth);
        vec3 delta = sample_position - position;

        float distance_sq = dot(delta, delta);

        if (distance_sq <= 0.00000001) {
            continue;
        }

        float closer = step(position.z + bias, sample_position.z);

        float range = 1.0 - smoothstep(
            radius_sq * 0.0324,
            radius_sq,
            distance_sq
        );

        float facing = saturate(
            dot(material.normal, delta) *
            inversesqrt(distance_sq) *
            0.5 + 0.5
        );

        occlusion += closer * range * facing;
        total += 1.0;
    }

    if (total <= 0.0) {
        return 1.0;
    }

    return clamp(
        1.0 - occlusion * rcp_safe(total, 1.0),
        0.0,
        1.0
    );
}

void main() {
    bool ssao_enabled = params.ssao.x > 0.0;

    SurfaceMaterial material = SurfaceMaterial(
        vec3(0.0, 0.0, 1.0),
        1.0
    );

    if (ssao_enabled) {
        material = surface_material(frag_uv);
    }

    vec3 color = texture(scene_color, frag_uv).rgb;

    color = high_quality_fxaa_scene_color(
        frag_uv,
        color,
        material,
        ssao_enabled
    );

    if (ssao_enabled) {
        float ao = screen_space_ao(frag_uv, material);

        color *= mix(
            1.0 - params.ssao.x * 0.38,
            1.0,
            ao
        );
    }

    out_color = vec4(apply_camera_effects(color), 1.0);
}