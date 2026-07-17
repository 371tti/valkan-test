#include "slang_compat.slang"

layout(location = 1) in float3 frag_normal;
layout(location = 2) in float2 frag_uv;
layout(location = 3) in float4 frag_color;
layout(location = 4) in float4 frag_tangent;
layout(location = 9) in float3 frag_world_pos;
layout(location = 10) in float frag_view_depth;
layout(location = 0) out float4 out_color;

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
    float4 emissive_light_position_radius[4];
    float4 emissive_light_color[4];
    float4 emissive_light_direction_radius[4];
    float4 emissive_light_size_kind[4];
    float4 emissive_light_count;
};
[[vk::binding(0, 0)]] ConstantBuffer<FrameCamera> frame_camera;

struct MaterialParams {
    float4 base_color_factor;
    float4 emissive_occlusion;
    float4 pbr_alpha;
    uint4 flags;
};
[[vk::binding(0, 1)]] ConstantBuffer<MaterialParams> material;

[[vk::binding(1, 1)]] Sampler2D base_color_texture;
[[vk::binding(2, 1)]] Sampler2D normal_texture;
[[vk::binding(3, 1)]] Sampler2D metallic_roughness_texture;
[[vk::binding(4, 1)]] Sampler2D occlusion_texture;
[[vk::binding(5, 1)]] Sampler2D emissive_texture;
[[vk::binding(0, 2)]] Sampler2D shadow_cascade_0;
[[vk::binding(1, 2)]] Sampler2D shadow_cascade_1;
[[vk::binding(2, 2)]] Sampler2D shadow_cascade_2;
[[vk::binding(3, 2)]] Sampler2D shadow_cascade_3;
[[vk::binding(4, 2)]] Sampler2D translucent_shadow_0;
[[vk::binding(5, 2)]] Sampler2D translucent_shadow_1;
[[vk::binding(6, 2)]] Sampler2D translucent_shadow_2;
[[vk::binding(7, 2)]] Sampler2D translucent_shadow_3;
[[vk::binding(8, 2)]] Sampler2D raw_shadow_cascade_0;
[[vk::binding(9, 2)]] Sampler2D raw_shadow_cascade_1;
[[vk::binding(10, 2)]] Sampler2D raw_shadow_cascade_2;
[[vk::binding(11, 2)]] Sampler2D raw_shadow_cascade_3;
[[vk::binding(12, 2)]] SamplerCube local_shadow_depth[4];

static const uint TEX_BASE_COLOR = 1u << 0;
static const uint TEX_NORMAL = 1u << 1;
static const uint TEX_METALLIC_ROUGHNESS = 1u << 2;
static const uint TEX_OCCLUSION = 1u << 3;
static const uint TEX_EMISSIVE = 1u << 4;
#include "shadow_sampling.slang"
#include "pbr_lighting.slang"

bool has_texture(uint bit) {
    return (material.flags.y & bit) != 0u;
}

float4 apply_alpha(float4 base_color) {
    if (material.flags.x == 1 && base_color.a <= material.pbr_alpha.z) {
        discard;
    }
    if (material.flags.x == 2 && base_color.a <= 0.001) {
        discard;
    }
    return base_color;
}

float3 normal_from_derivatives(float3 vertex_normal, float3 tangent_space_normal) {
    float3 dp1 = ddx(frag_world_pos);
    float3 dp2 = ddy(frag_world_pos);
    float2 duv1 = ddx(frag_uv);
    float2 duv2 = ddy(frag_uv);

    float det = duv1.x * duv2.y - duv1.y * duv2.x;
    if (abs(det) < 0.000001) {
        return vertex_normal;
    }

    float inv_det = 1.0 / det;
    float3 tangent_raw = (dp1 * duv2.y - dp2 * duv1.y) * inv_det;
    if (dot(tangent_raw, tangent_raw) < 0.000001) {
        return vertex_normal;
    }

    float3 tangent = tangent_raw - vertex_normal * dot(vertex_normal, tangent_raw);
    tangent = normalize_fast(tangent);

    float3 bitangent = cross(vertex_normal, tangent);
    float3 bitangent_raw = (-dp1 * duv2.x + dp2 * duv1.x) * inv_det;
    if (dot(bitangent, bitangent_raw) < 0.0) {
        bitangent = -bitangent;
    }

    return normalize_fast(
        tangent * tangent_space_normal.x +
        bitangent * tangent_space_normal.y +
        vertex_normal * tangent_space_normal.z
    );
}

float3 normal_from_map(float3 vertex_normal, float4 vertex_tangent) {
    float3 map = sample_texture(normal_texture, frag_uv).xyz * 2.0 - 1.0;
    map.xy *= material.pbr_alpha.w;
    float3 tangent_space_normal = normalize_fast(map);

    float3 tangent = vertex_tangent.xyz;
    float tangent_len_sq = dot(tangent, tangent);
    if (tangent_len_sq < 0.000001) {
        return normal_from_derivatives(vertex_normal, tangent_space_normal);
    }

    tangent = tangent - vertex_normal * dot(vertex_normal, tangent);
    tangent_len_sq = dot(tangent, tangent);
    if (tangent_len_sq < 0.000001) {
        return normal_from_derivatives(vertex_normal, tangent_space_normal);
    }
    tangent *= rsqrt(tangent_len_sq);

    float tangent_sign = vertex_tangent.w < 0.0 ? -1.0 : 1.0;
    float3 bitangent = cross(vertex_normal, tangent) * tangent_sign;
    return normalize_fast(
        tangent * tangent_space_normal.x +
        bitangent * tangent_space_normal.y +
        vertex_normal * tangent_space_normal.z
    );
}

float3 surface_normal(bool front_facing) {
    float3 normal = normalize_fast(frag_normal);
    if (material.flags.z != 0u && !front_facing) {
        normal = -normal;
    }
    if (has_texture(TEX_NORMAL)) {
        normal = normal_from_map(normal, frag_tangent);
    }
    return normal;
}

float4 read_base_color() {
    float4 base_color = frag_color * material.base_color_factor;
    if (has_texture(TEX_BASE_COLOR)) {
        base_color *= sample_texture(base_color_texture, frag_uv);
    }
    return apply_alpha(base_color);
}

void read_pbr(out float metallic, out float roughness) {
    metallic = clamp(material.pbr_alpha.x, 0.0, 1.0);
    roughness = clamp(material.pbr_alpha.y, 0.04, 1.0);
    if (has_texture(TEX_METALLIC_ROUGHNESS)) {
        float4 mr = sample_texture(metallic_roughness_texture, frag_uv);
        roughness = clamp(roughness * mr.g, 0.04, 1.0);
        metallic = clamp(metallic * mr.b, 0.0, 1.0);
    }
}

float read_occlusion() {
    if (
        frame_camera.ambient_color.w <= 0.000001 &&
        dot(frame_camera.ambient_color.rgb, frame_camera.ambient_color.rgb) <= 0.000001
    ) {
        return 1.0;
    }
    if (!has_texture(TEX_OCCLUSION)) {
        return 1.0;
    }
    float strength = clamp(material.emissive_occlusion.a, 0.0, 1.0);
    if (strength <= 0.0) {
        return 1.0;
    }
    return lerp(1.0, sample_texture(occlusion_texture, frag_uv).r, strength);
}

float3 read_emissive() {
    float3 emissive = material.emissive_occlusion.rgb;
    if (dot(emissive, emissive) <= 0.0) {
        return float3(0.0);
    }
    if (has_texture(TEX_EMISSIVE)) {
        emissive *= sample_texture(emissive_texture, frag_uv).rgb;
    }
    return emissive;
}

[shader("fragment")]
void main(bool front_facing : SV_IsFrontFace) {
    float4 base_color = read_base_color();
    if (scene_lighting_disabled()) {
        out_color = float4(read_emissive(), base_color.a);
        return;
    }
    float metallic;
    float roughness;
    read_pbr(metallic, roughness);
    float3 normal = surface_normal(front_facing);
    out_color = float4(
        shade_pbr(
            base_color.rgb,
            normal,
            metallic,
            roughness,
            read_occlusion(),
            read_emissive(),
            base_color.a
        ),
        base_color.a
    );
}
