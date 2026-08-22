#include "slang_compat.slang"
#include "temporal_reprojection.slang"

layout(location = 0) in float2 frag_uv;
layout(location = 0) out float4 out_history_color;
layout(location = 1) out float2 out_motion_vector;
layout(location = 2) out float out_linear_depth;
layout(location = 3) out float4 out_normal_history;

[[vk::binding(0, 0)]] Sampler2D current_color;
[[vk::binding(1, 0)]] Sampler2D current_depth;
[[vk::binding(2, 0)]] Sampler2D current_normal;
[[vk::binding(3, 0)]] Sampler2D previous_history_color;
[[vk::binding(4, 0)]] Sampler2D previous_linear_depth;
[[vk::binding(5, 0)]] Sampler2D previous_normal;
[[vk::binding(7, 0)]] Sampler2D current_transparent_normal;

struct TemporalResolveParams {
    float4x4 current_view_projection;
    float4x4 inverse_current_view_projection;
    float4x4 previous_view_projection;
    float4x4 inverse_current_view;
    float4x4 inverse_previous_view;
    // xy = texel size, z = maximum history feedback, w = reset history.
    float4 texel_feedback_reset;
    // x = absolute depth threshold, y = relative threshold, z = normal cosine, w = sample limit.
    float4 rejection;
    // xy = current pixel jitter, zw = previous pixel jitter (diagnostics/shared consumers).
    float4 jitter_pixels;
};
[[vk::binding(6, 0)]] ConstantBuffer<TemporalResolveParams> temporal;

float3 rgb_to_ycocg(float3 rgb) {
    return float3(
        rgb.r * 0.25 + rgb.g * 0.5 + rgb.b * 0.25,
        rgb.r * 0.5 - rgb.b * 0.5,
        -rgb.r * 0.25 + rgb.g * 0.5 - rgb.b * 0.25
    );
}

float3 ycocg_to_rgb(float3 ycocg) {
    float temporary = ycocg.x - ycocg.z;
    return float3(
        temporary + ycocg.y,
        ycocg.x + ycocg.z,
        temporary - ycocg.y
    );
}

void current_neighborhood_bounds(
    float3 center_value,
    out float3 minimum_value,
    out float3 maximum_value,
    out float3 standard_deviation
) {
    int2 dimensions = texture_dimensions(current_color, 0);
    int2 pixel = clamp(int2(frag_uv * float2(dimensions)), int2(0), dimensions - int2(1));
    minimum_value = center_value;
    maximum_value = center_value;
    float3 sum = center_value;
    float3 sum_squared = center_value * center_value;

    [unroll]
    for (int y = -1; y <= 1; ++y) {
        [unroll]
        for (int x = -1; x <= 1; ++x) {
            if (x == 0 && y == 0) {
                continue;
            }
            int2 coordinate = clamp(pixel + int2(x, y), int2(0), dimensions - int2(1));
            float2 sample_uv = (float2(coordinate) + float2(0.5)) / float2(dimensions);
            float3 sample_value = rgb_to_ycocg(sample_texture(current_color, sample_uv).rgb);
            minimum_value = min(minimum_value, sample_value);
            maximum_value = max(maximum_value, sample_value);
            sum += sample_value;
            sum_squared += sample_value * sample_value;
        }
    }

    float3 mean = sum / 9.0;
    float3 variance = max(sum_squared / 9.0 - mean * mean, float3(0.0));
    standard_deviation = sqrt(variance);

    // Intersect the hard neighborhood range with a variance box. This rejects isolated temporal
    // outliers while retaining the wider distribution around genuine geometry edges.
    float3 extent = maximum_value - minimum_value;
    float3 expansion = max(extent * 0.08, float3(0.0005, 0.001, 0.001));
    minimum_value -= expansion;
    maximum_value += expansion;
    float3 variance_radius =
        standard_deviation * float3(1.5, 2.0, 2.0) + float3(0.0005, 0.001, 0.001);
    minimum_value = max(minimum_value, mean - variance_radius);
    maximum_value = min(maximum_value, mean + variance_radius);
}

float current_linear_depth(float3 world_position) {
    float4 clip = mul(temporal.current_view_projection, float4(world_position, 1.0));
    return abs(clip.w);
}

[shader("fragment")]
void main() {
    // Stable CSM + PCSS is resolved directly into the HDR scene color. Reading that target here
    // keeps the mandatory 3x3 TAA clamp at one color fetch per neighbor.
    float4 current_sample = sample_texture(current_color, frag_uv);
    float device_depth = sample_texture(current_depth, frag_uv).r;
    float4 normal_meta = sample_texture(current_normal, frag_uv);
    bool has_transparent_surface =
        sample_texture(current_transparent_normal, frag_uv).w > 0.5;

    float3 world_position;
    float2 matrix_previous_uv;
    float expected_previous_depth;
    bool reprojected = temporal_reproject_world(
        frag_uv,
        device_depth,
        temporal.inverse_current_view_projection,
        temporal.previous_view_projection,
        world_position,
        matrix_previous_uv,
        expected_previous_depth
    );

    float2 previous_uv = temporal_stable_previous_uv(
        matrix_previous_uv,
        temporal.jitter_pixels.xy,
        temporal.jitter_pixels.zw,
        temporal.texel_feedback_reset.xy
    );
    reprojected = reprojected && temporal_uv_valid(previous_uv);

    float2 motion = reprojected ? previous_uv - frag_uv : float2(0.0);
    out_motion_vector = motion;
    out_linear_depth = current_linear_depth(world_position);
    // xy remains the shared opaque octahedral normal. The alpha channel carries transparent
    // coverage so appearing/disappearing transparent surfaces never reuse the opposite class.
    out_normal_history = float4(normal_meta.xyz, has_transparent_surface ? 1.0 : 0.0);

    bool reset_history = temporal.texel_feedback_reset.w > 0.5;
    // Transparent geometry does not own the shared depth history, so it uses the opaque receiver
    // behind it for a conservative camera reprojection. Coverage class matching below prevents
    // appearing/disappearing layers from leaking color; a reactive blend limits motion trails.
    bool history_valid = reprojected && !reset_history;
    float4 previous_sample = float4(0.0);
    float history_confidence = 0.0;
    int2 previous_surface_coordinate = int2(0);

    if (history_valid) {
        history_valid = temporal_best_previous_depth_sample(
            previous_linear_depth,
            previous_uv,
            expected_previous_depth,
            temporal.rejection.x,
            temporal.rejection.y,
            previous_surface_coordinate,
            history_confidence
        );
    }

    float4 previous_normal_meta = float4(0.0);
    if (history_valid) {
        previous_normal_meta = load_texel(previous_normal, previous_surface_coordinate, 0);
        bool previous_has_transparent_surface = previous_normal_meta.w > 0.5;
        history_valid = previous_has_transparent_surface == has_transparent_surface;
    }

    // Background pixels have no meaningful surface normal. Depth reprojection is sufficient there.
    if (history_valid && device_depth < 0.9999) {
        float3 current_normal_value = normalize(mul(
            temporal.inverse_current_view,
            float4(temporal_decode_octahedral_normal(normal_meta.xy), 0.0)
        ).xyz);
        float3 previous_normal_value = normalize(mul(
            temporal.inverse_previous_view,
            float4(temporal_decode_octahedral_normal(
                previous_normal_meta.xy
            ), 0.0)
        ).xyz);
        history_confidence *= temporal_normal_confidence(
            current_normal_value,
            previous_normal_value,
            temporal.rejection.z
        );
        history_valid = history_confidence > 0.0;
    }

    float history_limit = max(temporal.rejection.w, 1.0);
    if (!history_valid) {
        out_history_color = float4(max(current_sample.rgb, float3(0.0)), 1.0 / history_limit);
        return;
    }

    previous_sample = sample_texture(previous_history_color, previous_uv);
    // RGB remains bilinear for smooth motion, while the accumulation count must come from the
    // exact depth/normal surface chosen above instead of blending across a silhouette.
    previous_sample.a = load_texel(
        previous_history_color,
        previous_surface_coordinate,
        0
    ).a;
    float3 neighborhood_minimum;
    float3 neighborhood_maximum;
    float3 neighborhood_standard_deviation;
    current_neighborhood_bounds(
        rgb_to_ycocg(current_sample.rgb),
        neighborhood_minimum,
        neighborhood_maximum,
        neighborhood_standard_deviation
    );
    float3 clamped_history = ycocg_to_rgb(clamp(
        rgb_to_ycocg(max(previous_sample.rgb, float3(0.0))),
        neighborhood_minimum,
        neighborhood_maximum
    ));

    float previous_count =
        clamp(previous_sample.a, 0.0, 1.0) * history_limit * saturate(history_confidence);
    float new_count = min(previous_count + 1.0, history_limit);
    float minimum_current_weight = 1.0 - clamp(temporal.texel_feedback_reset.z, 0.0, 0.98);
    float current_weight = max(1.0 / max(new_count, 1.0), minimum_current_weight);
    if (has_transparent_surface) {
        // No object velocity/transparent depth history exists yet. Retain enough history to
        // suppress jitter, but keep a deliberately reactive floor to bound trails during motion.
        current_weight = max(current_weight, 0.35);
    }

    // Fade history continuously near depth/normal rejection thresholds. Hard binary rejection at
    // a silhouette makes the effective blend alternate from one frame to the next.
    current_weight = max(current_weight, 1.0 - saturate(history_confidence));
    float current_luma = dot(current_sample.rgb, float3(0.2126, 0.7152, 0.0722));
    float history_luma = dot(clamped_history, float3(0.2126, 0.7152, 0.0722));
    float relative_luma_delta = abs(current_luma - history_luma) / max(current_luma, 0.25);
    float edge_variance = saturate(neighborhood_standard_deviation.x / max(current_luma, 0.10));
    float responsive_luma_weight =
        clamp(relative_luma_delta * 0.08, 0.0, 0.10) * (1.0 - edge_variance);
    current_weight = max(current_weight, responsive_luma_weight);

    float3 resolved = max(
        lerp(clamped_history, max(current_sample.rgb, float3(0.0)), current_weight),
        float3(0.0)
    );
    out_history_color = float4(resolved, new_count / history_limit);
}
