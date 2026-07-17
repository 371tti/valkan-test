#include "slang_compat.slang"

layout(location = 0) in float3 in_position;
layout(location = 2) in float2 in_uv;
layout(location = 3) in float4 in_color;

layout(location = 0) out float2 frag_uv;
layout(location = 1) out float4 frag_color;

struct FrameCamera {
    float4x4 view_proj;
    float4x4 view;
    float4x4 shadow_view_proj[4];
    float4 shadow_cascade_splits;
    float4 shadow_cascade_texel_world;
    float4 shadow_cascade_depth_span;
    float4 camera_pos;
    float4 light_dir;
    float4 light_color;
    float4 ambient_color;
    float4 contact_shadow;
    float4x4 local_shadow_view_proj[24];
    float4 local_shadow_params[4];
};
[[vk::binding(0, 0)]] ConstantBuffer<FrameCamera> frame_camera;

struct ShadowCascade {
    uint cascade_index;
};
[[vk::push_constant]] ConstantBuffer<ShadowCascade> shadow_cascade;

[shader("vertex")]
void main(out float4 out_position : SV_Position) {
    uint index = shadow_cascade.cascade_index;
    float4x4 shadow_view_proj = index >= 4u
        ? frame_camera.local_shadow_view_proj[min(index - 4u, 23u)]
        : frame_camera.shadow_view_proj[min(index, 3u)];
    frag_uv = in_uv;
    frag_color = in_color;
    out_position = mul(shadow_view_proj, float4(in_position, 1.0));
}
