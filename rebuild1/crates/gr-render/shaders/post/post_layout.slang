#pragma once

layout(location = 0) in float2 frag_uv;
layout(location = 0) out float4 out_color;

[[vk::binding(0, 2)]] Sampler2D scene_color;
[[vk::binding(1, 2)]] Sampler2D scene_depth;
[[vk::binding(2, 2)]] Sampler2D scene_normal_roughness;
[[vk::binding(3, 2)]] Sampler2D scene_transparent_normal_roughness;
[[vk::binding(4, 2)]] Sampler2D bloom_texture;
[[vk::binding(5, 2)]] Sampler2D god_ray_texture_0;
[[vk::binding(6, 2)]] Sampler2D god_ray_texture_1;

struct PostParams {
    float4 white_balance;
    float4 camera;
    float4 depth;
    float4 ssao;
    float4 ssr;
    float4 aa;
    float4 bloom;
    float4 features;
    float4 god_ray_source0;
    float4 god_ray_color0;
    float4 god_ray_source1;
    float4 god_ray_color1;
    float exposure;
    float contrast;
    float saturation;
    float enabled;
};
[[vk::push_constant]] ConstantBuffer<PostParams> params;
