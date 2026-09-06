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
    // x = current color is the volumetric resolve; y = volumetric history limit;
    // z = continuous directional-light change signal for the volumetric history response.
    // Volumetric history is anchored by the current scene-depth endpoint when one exists.
    float4 effects;
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

    bool volumetric_input = temporal.effects.x > 0.5;
    float3 world_position = float3(0.0);
    // Keep the outputs deterministic even when depth reconstruction rejects the current sample.
    // `temporal_reproject_world` also writes these values on its failure paths, but explicit
    // initialization prevents an invalid/optimized-out path from feeding undefined coordinates
    // into the stable fallback calculation on drivers that evaluate both expressions eagerly.
    float2 matrix_previous_uv = frag_uv;
    float expected_previous_depth = 0.0;
    float current_surface_depth = 0.0;
    bool reprojected = false;
    // The matrix projection is the history coordinate of the same world point. Keep it as the
    // primary lookup: removing the jitter delta here would force a sloped/alpha-tested surface to
    // read a different previous pixel and its depth test would then toggle with every Halton
    // phase. The stable-grid coordinate remains a fallback below for the rare case where the exact
    // reprojected footprint has no compatible depth tap.
    float2 previous_uv = frag_uv;
    float2 stable_previous_uv = frag_uv;
    bool current_surface_valid = temporal_reproject_world(
            frag_uv,
            device_depth,
            temporal.inverse_current_view_projection,
            temporal.previous_view_projection,
            world_position,
            matrix_previous_uv,
            expected_previous_depth
        );
    stable_previous_uv = temporal_stable_previous_uv(
        matrix_previous_uv,
        temporal.jitter_pixels.xy,
        temporal.jitter_pixels.zw,
        temporal.texel_feedback_reset.xy
    );
    previous_uv = matrix_previous_uv;
    reprojected = current_surface_valid && temporal_uv_valid(previous_uv);

    float2 jitter_motion = (
        temporal.jitter_pixels.zw - temporal.jitter_pixels.xy
    ) * temporal.texel_feedback_reset.xy;
    float2 motion = reprojected ? previous_uv - frag_uv : float2(0.0);
    // Motion vectors describe camera/object motion, not the deliberate sub-pixel sample phase.
    // Consumers that use this attachment therefore do not see a one-pixel velocity alternating
    // with every Halton sample on an otherwise stationary surface.
    float2 camera_motion = motion - jitter_motion;
    out_motion_vector = camera_motion;
    // Keep all TAA history attachments on the same current-frame raster.  The output pass itself
    // is unjittered, but its samples are taken from the jittered scene at `frag_uv`; post effects
    // undo that sample offset when they query the history.  Ordinary color must still publish the
    // current surface depth here -- leaving this at the volumetric-only zero default makes every
    // opaque history lookup fall back to the jittered scene depth and makes its rejection toggle.
    if (current_surface_valid) {
        current_surface_depth = current_linear_depth(world_position);
    }
    out_linear_depth = current_surface_depth;
    // xy remains the shared opaque octahedral normal and z keeps roughness.  The R8 history target
    // has only one remaining channel, so pack material reflectance and the transparent class into
    // disjoint alpha ranges.  A raw 0/1 class flag (the former representation) destroyed
    // reflectance for post SSR and also made every highly reflective opaque pixel look transparent
    // once the stable metadata route consumed this attachment.
    float normal_reflectance = saturate(normal_meta.w);
    float normal_history_alpha = has_transparent_surface
        ? 0.52 + normal_reflectance * 0.48
        : normal_reflectance * 0.48;
    out_normal_history = float4(
        normal_meta.xyz,
        normal_history_alpha
    );

    bool reset_history = temporal.texel_feedback_reset.w > 0.5;
    // Transparent geometry does not own the shared depth history, so ordinary scene color uses the
    // opaque receiver behind it for a conservative camera reprojection. Volumetric color uses the
    // endpoint-reprojected ray and the current/previous surface metadata to reject a moving layer.
    bool history_valid = reprojected && !reset_history;
    float4 previous_sample = float4(0.0);
    float history_confidence = volumetric_input ? 1.0 : 0.0;
    int2 previous_surface_coordinate = int2(0);
    // Remove only the known Halton phase from the projected motion before deciding whether this
    // is a stationary surface. A near-zero residual means that only the current/previous sample
    // phase differs; real camera motion remains above this guard.
    float motion_pixels = reprojected
        ? length(camera_motion / max(temporal.texel_feedback_reset.xy, float2(1.0e-6)))
        : 0.0;
    // Matrix reprojection contains sub-pixel noise even for a stationary camera (depth is
    // reconstructed from a quantized, jittered raster).  Treat the whole sub-pixel envelope as
    // jitter-only; otherwise a pixel that lands within roughly half a texel of a silhouette
    // alternates between the depth/normal rejection path and the accumulation path every Halton
    // phase.  Real camera motion is measured in output pixels and remains above this guard.
    bool jitter_only = reprojected && motion_pixels < 0.75;

    if (history_valid) {
        // Do not truncate the reprojected UV to one history texel.  A Halton phase can put
        // the stable ray on either side of a pixel edge; a single integer load then alternates
        // between foreground and background depth and makes the history visibly pulse. Rebuild
        // both the ordinary and volumetric histories from the same depth-aware footprint so a
        // matching surface stays continuous. Only a footprint with no compatible depth is
        // rejected; an individual tap crossing the threshold cannot toggle the whole history.
        previous_sample = temporal_depth_aware_history_sample(
            previous_history_color,
            previous_linear_depth,
            previous_uv,
            expected_previous_depth,
            temporal.rejection.x,
            temporal.rejection.y,
            jitter_only,
            previous_surface_coordinate,
            history_confidence
        );
        if (history_confidence < 0.0 && temporal_uv_valid(stable_previous_uv)) {
            // A thin silhouette can leave no compatible tap at the exact same-world UV because
            // the previous raster classified the neighboring texels differently. Retry once on
            // the stable output coordinate; this is a recovery path, never the normal lookup.
            previous_sample = temporal_depth_aware_history_sample(
                previous_history_color,
                previous_linear_depth,
                stable_previous_uv,
                expected_previous_depth,
                temporal.rejection.x,
                temporal.rejection.y,
                jitter_only,
                previous_surface_coordinate,
                history_confidence
            );
            if (history_confidence >= 0.0) {
                previous_uv = stable_previous_uv;
                motion = previous_uv - frag_uv;
                camera_motion = motion - jitter_motion;
                motion_pixels = length(
                    camera_motion / max(temporal.texel_feedback_reset.xy, float2(1.0e-6))
                );
                jitter_only = motion_pixels < 0.75;
            }
        }
        // Zero confidence is still a geometrically compatible footprint.  It means the depth is
        // at the soft edge of the rejection band and should make this pixel fully current-frame,
        // not alternate between the valid and invalid branches as the jitter phase changes.  The
        // helper uses a negative sentinel only when no tap belongs to the reprojected surface.
        history_valid = history_confidence >= 0.0;
    }

    float4 previous_normal_meta = float4(0.0);
    if (history_valid && !volumetric_input && !jitter_only) {
        previous_normal_meta = load_texel(previous_normal, previous_surface_coordinate, 0);
        bool previous_has_transparent_surface = previous_normal_meta.w > 0.5;
        if (previous_has_transparent_surface != has_transparent_surface) {
            // Treat a class change as a continuous reactive signal.  The previous color is still
            // clamped below, but current_weight becomes one through (1 - confidence), avoiding a
            // hard branch flip when an alpha-tested edge changes class by one jittered sample.
            history_confidence = 0.0;
        }
    }

    // Background pixels have no meaningful surface normal. Depth reprojection is sufficient there.
    if (history_valid && !volumetric_input && device_depth < 0.9999 && !jitter_only &&
        history_confidence > 0.001) {
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
    }

    // The depth/normal attachments are consumed by post effects and by the legacy GodRay mask
    // on the next pass. Publishing the raw jittered MRT metadata here would make an alpha-tested
    // edge alternate between two full-resolution texels even when the color history itself is
    // stable. For a high-confidence, camera-stationary reprojection, carry the matching previous
    // surface metadata forward to the stable output lattice. Real camera motion, disocclusion,
    // and transparent coverage changes continue to publish the current sample below.
    if (history_valid && jitter_only && history_confidence > 0.85) {
        float previous_depth = load_texel(
            previous_linear_depth,
            previous_surface_coordinate,
            0
        ).r;
        if (isfinite(previous_depth) && previous_depth > 0.0) {
            out_linear_depth = previous_depth;
        }

        if (!has_transparent_surface) {
            float4 previous_stable_normal = load_texel(
                previous_normal,
                previous_surface_coordinate,
                0
            );
            // Alpha above 0.5 identifies the transparent-class packing. Never propagate that
            // class into an opaque output pixel; its current normal/reflectance remains the safe
            // fallback for a coverage transition.
            if (previous_stable_normal.w <= 0.5) {
                out_normal_history = previous_stable_normal;
            }
        }
    }

    // Volumetric lighting is resolved from a finite lattice and intentionally has no independent
    // volumetric history.  The CPU supplies a longer limit in effects.y so the ordinary final
    // TAA can integrate the stable jitter sequence without freezing the first visible PCSS box
    // at the opaque-color 16-frame limit.  Opaque scene color keeps its profile-controlled limit.
    float history_limit = max(
        volumetric_input ? temporal.effects.y : temporal.rejection.w,
        1.0
    );
    if (!history_valid) {
        out_history_color = float4(max(current_sample.rgb, float3(0.0)), 1.0 / history_limit);
        return;
    }

    // The depth-aware helper returns the color and accumulation count from the same compatible
    // jittered-raster footprint.
    float3 neighborhood_minimum;
    float3 neighborhood_maximum;
    float3 neighborhood_standard_deviation;
    current_neighborhood_bounds(
        rgb_to_ycocg(current_sample.rgb),
        neighborhood_minimum,
        neighborhood_maximum,
        neighborhood_standard_deviation
    );
    if (volumetric_input) {
        // The volumetric lighting pass reconstructs a finite XY/Z lattice. Its PCSS edge can move as
        // the camera or Sun changes even though the scene endpoint is unchanged. Keep a very
        // small extra hysteresis band around the 3x3 box so that edge does not clamp the history
        // to alternating sides of the same shadow.
        float3 volumetric_margin = max(
            abs(neighborhood_maximum - neighborhood_minimum) * 0.04,
            float3(0.0005, 0.001, 0.001)
        );
        neighborhood_minimum -= volumetric_margin;
        neighborhood_maximum += volumetric_margin;
    }
    float3 history_ycocg = rgb_to_ycocg(max(previous_sample.rgb, float3(0.0)));
    float3 clamped_history = ycocg_to_rgb(clamp(
        history_ycocg,
        neighborhood_minimum,
        neighborhood_maximum
    ));
    // A stationary output pixel is intentionally sampled at several jitter phases.  On a flat
    // receiver the 3x3 current neighborhood often has zero variance because the PCSS tap pattern
    // happens to classify the whole neighborhood alike.  Clipping the previous shadow value to
    // that tiny box turns a one-frame shadow quantization change into a current-frame pass, so the
    // same pixel flickers instead of converging.  The depth-aware footprint above already removed
    // other surfaces; for an opaque, high-confidence jitter-only sample it is therefore safe to
    // retain the history's lighting value. Real camera motion and transparent/volumetric layers
    // continue to use the conservative neighborhood clamp.
    bool preserve_jitter_only_lighting = jitter_only &&
        !has_transparent_surface &&
        history_confidence > 0.85;
    if (preserve_jitter_only_lighting) {
        clamped_history = max(previous_sample.rgb, float3(0.0));
    }

    float previous_count =
        clamp(previous_sample.a, 0.0, 1.0) * history_limit * saturate(history_confidence);
    float new_count = min(previous_count + 1.0, history_limit);
    // Exponential moving average: with `r` as the history feedback, a sample from k frames ago
    // has weight `(1-r) * r^k`.  This avoids the visible hard window/plateau produced by a uniform
    // count-based average while preserving the profile-controlled convergence rate.
    float history_feedback = clamp(temporal.texel_feedback_reset.z, 0.0, 0.995);
    float current_weight = 1.0 - history_feedback;
    if (has_transparent_surface && !jitter_only) {
        // No object velocity/transparent depth history exists yet. Retain enough history to
        // suppress motion trails, but do not apply a binary floor to a static Halton phase. At a
        // transparent silhouette the raster coverage can flip between neighboring pixels solely
        // because of jitter; forcing 35% current weight there makes that phase visible as a pixel
        // pulse even though the resolved coverage should converge normally.
        current_weight = max(current_weight, 0.35);
    }

    // Fade history continuously near depth/normal rejection thresholds. Hard binary rejection at
    // a silhouette makes the effective blend alternate from one frame to the next.
    current_weight = max(current_weight, 1.0 - saturate(history_confidence));
    float current_luma = dot(current_sample.rgb, float3(0.2126, 0.7152, 0.0722));
    float history_luma = dot(clamped_history, float3(0.2126, 0.7152, 0.0722));
    float relative_luma_delta = abs(current_luma - history_luma) / max(current_luma, 0.25);
    float edge_variance = saturate(neighborhood_standard_deviation.x / max(current_luma, 0.10));
    float response_scale = volumetric_input ? 0.025 : 0.08;
    float response_limit = volumetric_input ? 0.04 : 0.10;
    float response_edge_factor = volumetric_input ? 0.5 : (1.0 - edge_variance);
    float responsive_luma_weight =
        clamp(relative_luma_delta * response_scale, 0.0, response_limit) * response_edge_factor;
    current_weight = max(current_weight, responsive_luma_weight);
    if (volumetric_input) {
        // World-endpoint reprojection keeps the shaft attached while the camera moves.  Increase
        // the current contribution continuously with projected motion so a large pan cannot drag
        // an old CSM silhouette across the screen, while sub-pixel Halton jitter still receives
        // the full long-history average.
        float motion_response = smoothstep(0.75, 6.0, motion_pixels) * 0.35;
        current_weight = max(current_weight, motion_response);

        // A Sun/CSM change is not represented by the camera endpoint motion above. Without this
        // term, the long volumetric history keeps old shadow silhouettes for dozens of frames and
        // presents them as nested rectangular GodRay boxes. The CPU signal is continuous between
        // the coarser history-signature resets: static lighting keeps the long convergence, while
        // a moving light immediately gives the current shadow field enough authority.
        float light_change = saturate(temporal.effects.z);
        // A continuously orbiting Sun must not turn TAA into a current-frame pass.  The light
        // signal is therefore only a small override for a genuinely abrupt change; ordinary
        // sub-milliradian motion stays on the exponential history coefficient above.
        float light_response = smoothstep(0.45, 1.0, light_change) * 0.18;
        current_weight = max(current_weight, light_response);
    }

    // Opaque scene color contains the same animated directional-light/CSM field as the volumetric
    // route.  A day/night orbit changes the light by a small amount every frame; feeding that
    // signal directly into the current weight makes every PCSS tap visible as a pixel pulse and
    // effectively shortens the requested slow TAA convergence.  Keep the response reserved for an
    // actual light edit/cut.  Sub-pixel Sun motion is intentionally absorbed by the EMA, while a
    // large change still becomes current-frame dominant.
    if (!volumetric_input) {
        float light_change = saturate(temporal.effects.z);
        float light_response = smoothstep(0.55, 1.0, light_change) * 0.08;
        current_weight = max(current_weight, light_response);
    }

    float3 resolved = max(
        lerp(clamped_history, max(current_sample.rgb, float3(0.0)), current_weight),
        float3(0.0)
    );
    out_history_color = float4(resolved, new_count / history_limit);
}
