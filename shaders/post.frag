#version 450
#extension GL_GOOGLE_include_directive : require

#define DEBUG_DEFAULT 0.0
#define DEBUG_NO_TEXTURE 5.0

#include "scene_uniforms.glsl"
#include "common.glsl"

layout(set = 0, binding = 4) uniform sampler2D scene_color;
layout(set = 0, binding = 5) uniform sampler2D scene_depth;

layout(location = 0) in vec2 post_uv;
layout(location = 0) out vec4 out_color;

vec2 clamp_screen_uv(vec2 uv) {
    return clamp(uv, vec2(0.0015), vec2(0.9985));
}

float scene_depth_at(vec2 uv) {
    return texture(scene_depth, clamp_screen_uv(uv)).r;
}

vec3 bright_sample(vec2 uv) {
    vec3 color = texture(scene_color, clamp_screen_uv(uv)).rgb;
    float peak = max(max(color.r, color.g), color.b);
    float gate = smoothstep(0.78, 0.98, luma(color)) * smoothstep(0.84, 1.0, peak);

    return color * gate;
}

vec3 soft_bloom(vec2 uv, vec2 texel) {
    const vec2 taps[8] = vec2[](
        vec2(1.0, 0.0), vec2(-1.0, 0.0), vec2(0.0, 1.0), vec2(0.0, -1.0),
        vec2(0.72, 0.72), vec2(-0.72, 0.72), vec2(0.72, -0.72), vec2(-0.72, -0.72)
    );

    vec3 bloom = bright_sample(uv) * 0.28;
    for (int i = 0; i < 8; i++) {
        float radius = i < 4 ? 4.0 : 9.0;
        bloom += bright_sample(uv + taps[i] * texel * radius) * (i < 4 ? 0.12 : 0.08);
    }

    return bloom * 0.16;
}

vec3 view_ray(vec2 uv) {
    vec2 ndc = uv * 2.0 - 1.0;
    float aspect = scene.post_params.x;
    float tan_half_fov = scene.post_params.y;
    vec3 right = scene.camera_basis_x.xyz;
    vec3 up = scene.camera_basis_y.xyz;
    vec3 forward = scene.camera_basis_z.xyz;

    return normalize(
        forward +
        right * ndc.x * aspect * tan_half_fov -
        up * ndc.y * tan_half_fov
    );
}

vec3 reconstruct_world(vec2 uv, float depth) {
    float linear = linear_depth(depth);
    vec3 ray = view_ray(uv);
    float forward_distance = max(dot(ray, scene.camera_basis_z.xyz), 0.001);

    return scene.camera_pos.xyz + ray * (linear / forward_distance);
}

vec3 depth_normal(vec2 uv, vec2 texel) {
    float d0 = scene_depth_at(uv);
    vec3 center = reconstruct_world(uv, d0);
    vec3 right = reconstruct_world(uv + vec2(texel.x, 0.0), scene_depth_at(uv + vec2(texel.x, 0.0))) - center;
    vec3 left = center - reconstruct_world(uv - vec2(texel.x, 0.0), scene_depth_at(uv - vec2(texel.x, 0.0)));
    vec3 up = reconstruct_world(uv + vec2(0.0, texel.y), scene_depth_at(uv + vec2(0.0, texel.y))) - center;
    vec3 down = center - reconstruct_world(uv - vec2(0.0, texel.y), scene_depth_at(uv - vec2(0.0, texel.y)));
    vec3 dx = dot(right, right) < dot(left, left) ? right : left;
    vec3 dy = dot(up, up) < dot(down, down) ? up : down;
    vec3 normal = normalize(cross(dx, dy));

    return dot(normal, -view_ray(uv)) < 0.0 ? -normal : normal;
}

float weak_ssao(vec2 uv, vec3 world_pos, vec3 normal, vec2 texel) {
    const vec2 taps[12] = vec2[](
        vec2(0.92, 0.18), vec2(-0.74, 0.42), vec2(0.38, -0.83),
        vec2(-0.28, -0.91), vec2(0.12, 0.66), vec2(-0.96, -0.08),
        vec2(0.61, 0.71), vec2(-0.52, -0.58), vec2(0.80, -0.45),
        vec2(-0.10, 0.98), vec2(0.33, 0.24), vec2(-0.36, 0.08)
    );

    float depth = linear_depth(scene_depth_at(uv));
    float radius = clamp(depth * 0.018, 0.08, 1.15);
    float pixel_radius = clamp(42.0 / max(depth, 1.0), 2.0, 18.0);
    float occlusion = 0.0;
    float weight_sum = 0.0;

    for (int i = 0; i < 12; i++) {
        vec2 sample_uv = uv + taps[i] * texel * pixel_radius * (1.0 + float(i % 3) * 0.42);
        float sample_depth = scene_depth_at(sample_uv);
        if (sample_depth >= 0.9999) {
            continue;
        }

        vec3 sample_pos = reconstruct_world(sample_uv, sample_depth);
        vec3 delta = sample_pos - world_pos;
        float distance = length(delta);
        float normal_term = saturate(dot(normal, normalize(delta)) * 1.45 - 0.12);
        float range = 1.0 - smoothstep(radius * 0.25, radius, distance);
        float depth_term = smoothstep(0.001, radius * 0.5, depth - linear_depth(sample_depth));
        float weight = range * normal_term;

        occlusion += depth_term * weight;
        weight_sum += weight;
    }

    float ao = 1.0 - scene.gi_params.z * saturate(occlusion / max(weight_sum, 0.001));

    return mix(1.0, ao, smoothstep(0.0, 0.08, 1.0 - normal.y * 0.5));
}

vec4 project_world(vec3 world_pos) {
    vec4 projected = scene.view_proj * vec4(world_pos, 1.0);
    projected.xyz /= max(projected.w, 0.0001);

    return projected;
}

vec3 auxiliary_ssr(vec2 uv, vec3 base_color, vec3 world_pos, vec3 normal) {
    float debug_mode = scene.debug_params.x;
    if (abs(debug_mode - DEBUG_DEFAULT) > 0.1 && abs(debug_mode - DEBUG_NO_TEXTURE) > 0.1) {
        return vec3(0.0);
    }

    vec3 view_dir = normalize(world_pos - scene.camera_pos.xyz);
    float facing = saturate(1.0 - abs(dot(normal, -view_dir)));
    float floor_bias = smoothstep(0.12, 0.72, normal.y * 0.5 + 0.5);
    float strength = scene.gi_params.w * facing * floor_bias;
    if (strength <= 0.001) {
        return vec3(0.0);
    }

    vec3 ray = normalize(reflect(view_dir, normal));
    float max_distance = mix(2.5, 18.0, facing);
    vec3 origin = world_pos + normal * 0.035;
    vec3 hit_color = vec3(0.0);
    float hit_weight = 0.0;

    const int SSR_STEPS = 14;
    for (int step = 1; step <= SSR_STEPS; step++) {
        float t = max_distance * float(step) / float(SSR_STEPS);
        vec3 ray_pos = origin + ray * t;
        vec4 projected = project_world(ray_pos);
        vec2 sample_uv = projected.xy * 0.5 + 0.5;

        if (projected.w <= 0.0 || any(lessThan(sample_uv, vec2(0.0))) || any(greaterThan(sample_uv, vec2(1.0)))) {
            break;
        }

        float buffer_depth = scene_depth_at(sample_uv);
        if (buffer_depth >= 0.9999) {
            continue;
        }

        float ray_depth = linear_depth(projected.z);
        float sample_depth = linear_depth(buffer_depth);
        float thickness = max(0.08, ray_depth * 0.015);

        if (ray_depth > sample_depth - thickness && ray_depth < sample_depth + thickness * 3.0) {
            vec2 edge = smoothstep(vec2(0.0), vec2(0.12), sample_uv)
                * (1.0 - smoothstep(vec2(0.88), vec2(1.0), sample_uv));
            hit_weight = edge.x * edge.y * (1.0 - float(step) / 22.0);
            hit_color = texture(scene_color, sample_uv).rgb;
            break;
        }
    }

    return max(hit_color - base_color * 0.08, vec3(0.0)) * hit_weight * strength;
}

void main() {
    vec4 color = texture(scene_color, clamp_screen_uv(post_uv));
    float debug_mode = scene.debug_params.x;
    bool post_enabled = abs(debug_mode - DEBUG_DEFAULT) <= 0.1 || abs(debug_mode - DEBUG_NO_TEXTURE) <= 0.1;
    vec2 texel = 1.0 / vec2(textureSize(scene_color, 0));
    vec3 bloom = post_enabled ? soft_bloom(post_uv, texel) : vec3(0.0);
    float depth = scene_depth_at(post_uv);
    if (depth >= 0.9999) {
        out_color = vec4(color.rgb + bloom, 1.0);
        return;
    }

    if (!post_enabled) {
        out_color = vec4(color.rgb, 1.0);
        return;
    }

    vec3 world_pos = reconstruct_world(post_uv, depth);
    vec3 normal = depth_normal(post_uv, texel);
    float ao = weak_ssao(post_uv, world_pos, normal, texel);
    vec3 ssr = auxiliary_ssr(post_uv, color.rgb, world_pos, normal);

    out_color = vec4(color.rgb * ao + ssr + bloom, 1.0);
}
