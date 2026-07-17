#include "slang_compat.slang"

layout(location = 0) in float3 in_position;
layout(location = 1) in float3 in_normal;
layout(location = 4) in float4 in_color;

layout(location = 1) out float3 frag_normal;
layout(location = 3) out float4 frag_color;
layout(location = 9) out float3 frag_world_pos;
layout(location = 10) out float frag_view_depth;

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
};
[[vk::binding(0, 0)]] ConstantBuffer<FrameCamera> frame_camera;

[shader("vertex")]
void main(out float4 out_position : SV_Position) {
    float4 world_pos = float4(in_position, 1.0);
    out_position = mul(frame_camera.view_proj, world_pos);
    frag_normal = in_normal;
    frag_color = in_color;
    frag_world_pos = world_pos.xyz;
    frag_view_depth = max(-mul(frame_camera.view, world_pos).z, 0.0);
}
