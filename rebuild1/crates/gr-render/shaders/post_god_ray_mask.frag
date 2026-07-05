#version 450
#extension GL_GOOGLE_include_directive : require

layout(location = 0) in vec2 frag_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D scene_color;
layout(set = 0, binding = 1) uniform sampler2D scene_depth;
layout(set = 0, binding = 2) uniform sampler2D scene_transparent_normal_roughness;

#include "post_god_ray_common.glsl"

float depth_open_visibility(float depth) {
    if (depth >= 0.999995) {
        return 1.0;
    }

    if (depth <= 0.0) {
        return 0.0;
    }

    return smoothstep(0.99992, 0.999985, depth);
}

float background_visibility(vec2 uv) {
    vec2 texel = params.target.xy * 0.72;
    float visibility = depth_open_visibility(raw_depth_at(scene_depth, uv));

    visibility = min(visibility, depth_open_visibility(raw_depth_at(scene_depth, uv + vec2( texel.x, 0.0))));
    visibility = min(visibility, depth_open_visibility(raw_depth_at(scene_depth, uv + vec2(-texel.x, 0.0))));
    visibility = min(visibility, depth_open_visibility(raw_depth_at(scene_depth, uv + vec2(0.0,  texel.y))));
    visibility = min(visibility, depth_open_visibility(raw_depth_at(scene_depth, uv + vec2(0.0, -texel.y))));

    return visibility;
}

float transparent_coverage(vec2 uv) {
    if (params.features.x <= 0.5) {
        return 1.0;
    }

    vec4 transparent = texture(scene_transparent_normal_roughness, clamp(uv, vec2(0.0), vec2(1.0)));
    float has_transparent = step(1.0, transparent.w);

    return mix(1.0, 0.62, has_transparent);
}

float local_source_visibility(vec2 uv) {
    vec3 color = max(texture(scene_color, clamp(uv, vec2(0.0), vec2(1.0))).rgb, vec3(0.0));
    float luma = luminance_of(color);
    float threshold = max(params.bloom.y * 0.72, 0.12);

    return smoothstep(threshold, threshold * 5.5, luma);
}

vec3 source_mask(vec2 uv, vec4 source, vec4 source_color) {
    float strength = source.w;
    if (strength <= 0.0) {
        return vec3(0.0);
    }

    float directional = clamp(source_color.a, 0.0, 1.0);
    float dist = god_ray_aspect_distance(uv, source.xy);
    float radius = max(source.z, 0.002);
    float disc = exp2(-dist * dist * rcp_safe(radius * radius, 0.00001) * 2.8);
    float sky = background_visibility(uv);
    float local = local_source_visibility(uv);
    float source_fade = god_ray_source_screen_fade(source.xy, directional);
    float coverage = transparent_coverage(uv);

    float directional_air = sky * source_fade * 0.075;
    float directional_core = sky * source_fade * disc * 0.030;
    float directional_mask = directional_air + directional_core;
    float local_mask = local * disc * source_fade;
    float mask = mix(local_mask, directional_mask, directional) * coverage;

    return min(source_color.rgb * mask * strength * params.bloom.w, vec3(1.2));
}

void main() {
    vec3 mask = source_mask(frag_uv, params.source0, params.color0);
    mask += source_mask(frag_uv, params.source1, params.color1);
    float depth = raw_depth_at(scene_depth, frag_uv);

    out_color = vec4(max(mask, vec3(0.0)), depth);
}
