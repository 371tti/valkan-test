#include "slang_compat.slang"

layout(location = 0) in float2 frag_uv;
layout(location = 1) in float4 frag_color;

layout(location = 0) out float4 out_transmittance;

struct MaterialParams {
    float4 base_color_factor;
    float4 emissive_occlusion;
    float4 pbr_alpha;
    uint4 flags;
};
[[vk::binding(0, 1)]] ConstantBuffer<MaterialParams> material;

// All four stable CSM cascades share one four-layer depth array.
[[vk::binding(10, 2)]] Sampler2DArray opaque_shadow_depth;

struct ShadowCascade {
    uint cascade_index;
};
[[vk::push_constant]] ConstantBuffer<ShadowCascade> shadow_cascade;

float opaque_shadow_depth_at_pixel(int2 coordinate) {
    uint matrix_index = shadow_cascade.cascade_index;
    int4 location = int4(coordinate, int(matrix_index), 0);
    return opaque_shadow_depth.Load(location).r;
}

bool translucent_is_in_front_of_opaque(float4 sv_position) {
    // Rasterized framebuffer coordinates are non-negative; integer conversion is floor here.
    int2 coordinate = int2(sv_position.xy);
    float opaque_depth = opaque_shadow_depth_at_pixel(coordinate);
    // A tiny tolerance keeps coplanar transparent decals from disappearing due to raster precision.
    return sv_position.z <= opaque_depth + 0.00001;
}

float3 translucent_transmittance(float3 color, float alpha) {
    color = clamp(color, float3(0.0), float3(1.0));
    alpha = clamp(alpha, 0.0, 1.0);

    return lerp(float3(1.0), color, alpha);
}

[shader("fragment")]
void main(float4 sv_position : SV_Position) {
    float alpha = clamp(
        frag_color.a * material.base_color_factor.a,
        0.0,
        1.0
    );

    if (alpha <= 0.001) {
        discard;
    }
    // Reject transparent fragments before the opaque-depth array fetch whenever possible.
    if (!translucent_is_in_front_of_opaque(sv_position)) {
        discard;
    }
    float3 color = frag_color.rgb * material.base_color_factor.rgb;
    out_transmittance = float4(
        translucent_transmittance(color, alpha),
        clamp(sv_position.z, 0.0, 1.0)
    );
}
