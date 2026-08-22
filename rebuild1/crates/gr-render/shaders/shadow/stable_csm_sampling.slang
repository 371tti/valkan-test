#pragma once

// Stable CSM + PCSS directional visibility.
//
// Stable CSM keeps the four orthographic receiver projections aligned to a texel grid. PCSS is
// evaluated spatially in the current frame: a small blocker search estimates the penumbra and a
// variable-radius percentage-closer filter resolves the final visibility. There is deliberately
// no directional sample cursor or shadow-history buffer in this path.

static const uint STABLE_CSM_CASCADE_COUNT = 4u;
static const uint STABLE_CSM_MAX_BLOCKER_SAMPLES = 16u;
static const uint STABLE_CSM_MAX_FILTER_SAMPLES = 32u;

float stable_csm_rcp_safe(float value, float floor_value) {
    return 1.0 / max(value, floor_value);
}

uint stable_csm_hash_u32(uint value) {
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    return value;
}

float stable_csm_hash(uint2 pixel, uint cascade_index) {
    // Hash integer raster pixels instead of continuous world coordinates. The old sin(dot())
    // seed repeated at sub-unit world-space periods, which made high-sample PCSS reveal stable
    // bands on large receivers. This integer mix has no short spatial period in the viewport.
    uint seed = pixel.x * 0x9e3779b9u;
    seed += pixel.y * 0x85ebca6bu;
    seed += (cascade_index + 1u) * 0xc2b2ae35u;
    uint mixed = stable_csm_hash_u32(seed);
    return float(mixed & 0x00ffffffu) * (1.0 / 16777216.0);
}

float2 stable_csm_disk_offset(uint index, float inverse_count, float rotation) {
    float sample = float(index) + 0.5;
    float radius = sqrt(sample * inverse_count);
    float angle = rotation + sample * 2.39996323;
    return float2(cos(angle), sin(angle)) * radius;
}

float4 stable_csm_project(
    float4x4 shadow_view_projection,
    float3 world_position,
    float3 receiver_normal,
    float receiver_ndotl,
    float texel_world,
    float normal_offset_scale,
    out float3 projected,
    out float2 uv
) {
    float grazing = 1.0 - receiver_ndotl;
    // Move the receiver in world space before projection. This reduces self-intersection in the
    // PCSS blocker search without inflating the final shadow footprint in UV space.
    float normal_offset = max(texel_world, 0.00001)
        * lerp(0.04, 0.24, grazing)
        * max(normal_offset_scale, 0.0);
    float3 biased_position = world_position + receiver_normal * normal_offset;
    float4 position = mul(shadow_view_projection, float4(biased_position, 1.0));
    projected = position.xyz * stable_csm_rcp_safe(position.w, 0.0001);
    uv = projected.xy * 0.5 + 0.5;
    return position;
}

bool stable_csm_projection_valid(float4 position, float3 projected, float2 uv) {
    return position.w > 0.0 &&
        all(float3(uv, projected.z) >= float3(0.0)) &&
        all(float3(uv, projected.z) <= float3(1.0));
}

float stable_csm_receiver_bias(
    float receiver_ndotl,
    float texel_world,
    float inverse_depth_span,
    float cascade_resolution,
    float2 receiver_plane_gradient,
    float constant_bias_scale,
    float slope_bias_scale,
    float receiver_plane_bias_scale
) {
    float grazing = 1.0 - receiver_ndotl;
    float texel_depth = max(texel_world, 0.00001) * inverse_depth_span;
    float ndotl = max(receiver_ndotl, 0.05);
    float slope = min(sqrt(max(1.0 - ndotl * ndotl, 0.0)) / ndotl, 8.0);
    // D16_UNORM quantization is a hard lower bound: even a perfectly flat receiver can land on
    // the adjacent representable depth after rasterization.  Keep two quantization steps in the
    // baseline so the bias does not collapse when the cascade footprint becomes very small.
    float d16_quantization_depth = 2.0 / 65535.0;
    float constant_bias = d16_quantization_depth
        + (0.20 + grazing * 0.45) * texel_depth * max(constant_bias_scale, 0.0);
    float slope_bias = slope * texel_depth * 0.85 * max(slope_bias_scale, 0.0);
    // The receiver plane gradient is depth-per-UV.  Convert its largest one-texel excursion to
    // normalized shadow depth so each PCSS tap is covered by the same error model as the center.
    float plane_gradient = max(
        abs(receiver_plane_gradient.x),
        abs(receiver_plane_gradient.y)
    );
    float receiver_plane_bias = plane_gradient
        * stable_csm_rcp_safe(cascade_resolution, 1.0)
        * 1.25
        * max(receiver_plane_bias_scale, 0.0);
    return clamp(
        constant_bias + slope_bias + receiver_plane_bias,
        0.000002,
        0.02
    );
}

// Estimates the receiver plane's light-space depth gradient in shadow-UV units.  Applying this
// gradient at each blocker/filter tap keeps a sloped receiver from being compared against one
// constant center depth, which otherwise turns regular PCSS taps into visible horizontal bands.
float2 stable_csm_receiver_plane_gradient(
    float2 uv,
    float receiver_depth,
    float receiver_plane_bias_scale
) {
    if (receiver_plane_bias_scale <= 0.0) {
        return float2(0.0);
    }
    float2 uv_dx = ddx(uv);
    float2 uv_dy = ddy(uv);
    float depth_dx = ddx(receiver_depth);
    float depth_dy = ddy(receiver_depth);
    float determinant = uv_dx.x * uv_dy.y - uv_dx.y * uv_dy.x;
    if (abs(determinant) <= 1.0e-7) {
        return float2(0.0);
    }

    float inverse_determinant = rcp(determinant);
    float2 gradient = float2(
        (depth_dx * uv_dy.y - depth_dy * uv_dx.y) * inverse_determinant,
        (uv_dx.x * depth_dy - uv_dy.x * depth_dx) * inverse_determinant
    );
    // A bad/degenerate derivative must never expand a PCSS kernel to infinity.  The cap is in
    // depth-per-UV units, so a one-texel offset remains a small correction even at grazing angles.
    float gradient_limit = 0.25 * receiver_plane_bias_scale;
    return clamp(gradient, -gradient_limit, gradient_limit);
}

float stable_csm_depth_sample(
    Sampler2DArray shadow_depth,
    float2 uv,
    uint cascade_index
) {
    return shadow_depth.SampleLevel(float3(uv, float(cascade_index)), 0.0).r;
}

float stable_csm_compare(
    Sampler2DArrayShadow shadow_depth,
    float2 uv,
    uint cascade_index,
    float receiver_depth
) {
    return shadow_depth.SampleCmpLevelZero(
        float3(uv, float(cascade_index)),
        receiver_depth
    );
}

float stable_csm_pcss_visibility(
    Sampler2DArray raw_shadow_depth,
    Sampler2DArrayShadow shadow_depth,
    uint cascade_index,
    float4x4 shadow_view_projection,
    float3 world_position,
    uint2 sample_pixel,
    float3 receiver_normal,
    float3 surface_to_light,
    float receiver_ndotl,
    float texel_world,
    float depth_span,
    float inverse_depth_span,
    float cascade_resolution,
    float4 pcss_params,
    float4 receiver_params,
    bool debug_filter_radius
) {
    float3 projected;
    float2 uv;
    float4 shadow_position = stable_csm_project(
        shadow_view_projection,
        world_position,
        receiver_normal,
        receiver_ndotl,
        texel_world,
        receiver_params.z,
        projected,
        uv
    );
    if (!stable_csm_projection_valid(shadow_position, projected, uv)) {
        return debug_filter_radius ? 0.0 : 1.0;
    }

    float2 receiver_plane_gradient = stable_csm_receiver_plane_gradient(
        uv,
        projected.z,
        receiver_params.w
    );
    float bias = stable_csm_receiver_bias(
        receiver_ndotl,
        texel_world,
        inverse_depth_span,
        cascade_resolution,
        receiver_plane_gradient,
        receiver_params.x,
        receiver_params.y,
        receiver_params.w
    );
    // Keep the comparison depth biased, but preserve the geometric receiver depth for penumbra
    // estimation. Mixing the two makes the blocker distance jump whenever the bias changes.
    float receiver_compare_depth = projected.z - bias;
    // Both the search footprint and the penumbra conversion use the same cascade metrics. Hoist
    // the reciprocals once so high-quality kernels do not redo scalar divisions per stage.
    float inverse_resolution = stable_csm_rcp_safe(cascade_resolution, 1.0);
    float cascade_width_world = max(texel_world * cascade_resolution, 0.001);
    float inverse_cascade_width = stable_csm_rcp_safe(cascade_width_world, 0.001);
    float angular_radius = max(pcss_params.z, 0.0);
    float rotation = stable_csm_hash(sample_pixel, cascade_index) * 6.2831853;
    uint blocker_count = min(
        uint(max(pcss_params.x, 1.0) + 0.5),
        STABLE_CSM_MAX_BLOCKER_SAMPLES
    );
    uint filter_count = min(
        uint(max(pcss_params.y, 1.0) + 0.5),
        STABLE_CSM_MAX_FILTER_SAMPLES
    );
    float inverse_blocker_count = stable_csm_rcp_safe(float(blocker_count), 1.0);
    float inverse_filter_count = stable_csm_rcp_safe(float(filter_count), 1.0);

    // The blocker search is deliberately compact. Its radius is tied to the light's angular
    // extent and the current cascade footprint, so it does not swim when the camera translates.
    float search_radius_uv = max(
        1.5 * inverse_resolution,
        angular_radius * 0.18
    );
    if (pcss_params.w > 0.5) {
        search_radius_uv = max(search_radius_uv, 2.5 * inverse_resolution);
    }

    float blocker_sum = 0.0;
    float blocker_receiver_sum = 0.0;
    float blocker_weight = 0.0;
    // D16 quantization and receiver bias both create a shallow uncertainty band around the
    // blocker test. Softly weighting that band prevents one binary tap entering/leaving the
    // search disk from producing a visible stripe in average_blocker_depth.
    float blocker_softness = max(bias, 4.0 / 65535.0);
    for (uint sample_index = 0u; sample_index < blocker_count; ++sample_index) {
        float2 offset = stable_csm_disk_offset(sample_index, inverse_blocker_count, rotation);
        float2 sample_offset_uv = offset * search_radius_uv;
        float sample_receiver_depth = projected.z + dot(
            receiver_plane_gradient,
            sample_offset_uv
        );
        float sample_depth = stable_csm_depth_sample(
            raw_shadow_depth,
            uv + sample_offset_uv,
            cascade_index
        );
        // The compare depth is the same receiver-plane estimate minus the center bias. Reuse the
        // estimate instead of evaluating its gradient dot product a second time for every blocker.
        float blocker_delta = sample_receiver_depth - bias - sample_depth;
        float sample_weight = smoothstep(0.0, blocker_softness, blocker_delta);
        if (sample_weight > 0.0) {
            blocker_sum += sample_depth * sample_weight;
            blocker_receiver_sum += sample_receiver_depth * sample_weight;
            blocker_weight += sample_weight;
        }
    }

    if (blocker_weight <= 0.25) {
        return debug_filter_radius ? 0.0 : 1.0;
    }

    float inverse_blocker_weight = rcp(blocker_weight);
    float average_blocker_depth = blocker_sum * inverse_blocker_weight;
    float average_blocker_receiver_depth = blocker_receiver_sum * inverse_blocker_weight;
    float blocker_receiver_delta = max(
        average_blocker_receiver_depth - average_blocker_depth,
        0.0
    );
    float penumbra_world = blocker_receiver_delta * depth_span * angular_radius;
    float filter_radius_uv = max(
        inverse_resolution,
        penumbra_world * inverse_cascade_width
    );
    if (debug_filter_radius) {
        return saturate(filter_radius_uv * cascade_resolution / 20.0);
    }

    float visibility = 0.0;
    for (uint sample_index = 0u; sample_index < filter_count; ++sample_index) {
        float2 offset = stable_csm_disk_offset(sample_index, inverse_filter_count, rotation + 0.37);
        float2 sample_offset_uv = offset * filter_radius_uv;
        float sample_receiver_depth = receiver_compare_depth + dot(
            receiver_plane_gradient,
            sample_offset_uv
        );
        visibility += stable_csm_compare(
            shadow_depth,
            uv + sample_offset_uv,
            cascade_index,
            sample_receiver_depth
        );
    }
    return saturate(visibility * inverse_filter_count);
}

float3 stable_csm_transmittance(
    uint cascade_index,
    Sampler2D translucent_shadow,
    float4x4 shadow_view_projection,
    float3 world_position,
    float3 receiver_normal,
    float3 surface_to_light,
    float receiver_ndotl,
    float texel_world,
    float inverse_depth_span,
    float normal_offset_scale
) {
    float3 projected;
    float2 uv;
    float4 shadow_position = stable_csm_project(
        shadow_view_projection,
        world_position,
        receiver_normal,
        receiver_ndotl,
        texel_world,
        normal_offset_scale,
        projected,
        uv
    );
    if (!stable_csm_projection_valid(shadow_position, projected, uv)) {
        return float3(1.0);
    }
    float receiver_depth = projected.z;
    float4 sample_value = translucent_shadow.SampleLevel(uv, 0.0);
    float depth_tolerance = max(texel_world, 0.00001) * inverse_depth_span * 2.0;
    return sample_value.a + depth_tolerance >= receiver_depth
        ? clamp(sample_value.rgb, float3(0.0), float3(1.0))
        : float3(1.0);
}

float3 stable_csm_sample_transmittance(
    uint cascade_index,
    Sampler2D translucent_shadow_0,
    Sampler2D translucent_shadow_1,
    Sampler2D translucent_shadow_2,
    Sampler2D translucent_shadow_3,
    float4x4 shadow_view_projection[4],
    float3 world_position,
    float3 receiver_normal,
    float3 surface_to_light,
    float receiver_ndotl,
    float4 cascade_texel_world,
    float inverse_depth_span,
    uint translucent_mask,
    float normal_offset_scale
) {
    if ((translucent_mask & (1u << cascade_index)) == 0u) {
        return float3(1.0);
    }
    if (cascade_index == 0u) {
        return stable_csm_transmittance(0u, translucent_shadow_0, shadow_view_projection[0], world_position, receiver_normal, surface_to_light, receiver_ndotl, cascade_texel_world.x, inverse_depth_span, normal_offset_scale);
    }
    if (cascade_index == 1u) {
        return stable_csm_transmittance(1u, translucent_shadow_1, shadow_view_projection[1], world_position, receiver_normal, surface_to_light, receiver_ndotl, cascade_texel_world.y, inverse_depth_span, normal_offset_scale);
    }
    if (cascade_index == 2u) {
        return stable_csm_transmittance(2u, translucent_shadow_2, shadow_view_projection[2], world_position, receiver_normal, surface_to_light, receiver_ndotl, cascade_texel_world.z, inverse_depth_span, normal_offset_scale);
    }
    return stable_csm_transmittance(3u, translucent_shadow_3, shadow_view_projection[3], world_position, receiver_normal, surface_to_light, receiver_ndotl, cascade_texel_world.w, inverse_depth_span, normal_offset_scale);
}

float stable_csm_cascade_transition_width(float split) {
    return max(split * 0.045, 1.0);
}

float3 shadow_factor(
    Sampler2DArray raw_shadow_depth,
    Sampler2DArrayShadow shadow_depth,
    Sampler2D translucent_shadow_0,
    Sampler2D translucent_shadow_1,
    Sampler2D translucent_shadow_2,
    Sampler2D translucent_shadow_3,
    float4x4 shadow_view_proj[4],
    float3 world_position,
    float2 sample_seed,
    float3 receiver_normal,
    float3 surface_to_light,
    float camera_depth,
    float4 cascade_splits,
    float4 cascade_texel_world,
    float4 cascade_depth_span,
        float4 cascade_resolution,
        float4 pcss_params,
    float4 receiver_params,
    float translucent_mask,
    out float opaque_visibility,
    out float3 transmittance
) {
    bool debug_filter_radius = frame_camera.debug_view.x > 2.5 && frame_camera.debug_view.x < 3.5;
    float fade_width = max(cascade_splits.w * 0.07, 2.0);
    if (camera_depth >= cascade_splits.w + fade_width) {
        opaque_visibility = debug_filter_radius ? 0.0 : 1.0;
        transmittance = float3(1.0);
        return float3(opaque_visibility);
    }
    uint2 sample_pixel = uint2(max(sample_seed, float2(0.0)));
    float receiver_ndotl = saturate(dot(receiver_normal, surface_to_light));

    uint cascade_index =
        uint(camera_depth >= cascade_splits.x) +
        uint(camera_depth >= cascade_splits.y) +
        uint(camera_depth >= cascade_splits.z);
    cascade_index = min(cascade_index, STABLE_CSM_CASCADE_COUNT - 1u);
    uint primary = cascade_index;
    uint secondary = cascade_index;
    float transition_t = 0.0;
    bool transition_active = false;
    // Every split owns a symmetric overlap region.  Select the lower/upper pair directly rather
    // than deriving a transition index from threshold counts (which could skip a pair exactly at
    // the split and blend the next two cascades instead).
    for (uint boundary = 0u; boundary < STABLE_CSM_CASCADE_COUNT - 1u; ++boundary) {
        float split = cascade_splits[boundary];
        float transition_width = stable_csm_cascade_transition_width(split);
        if (abs(camera_depth - split) < transition_width) {
            primary = boundary;
            secondary = boundary + 1u;
            transition_t = saturate(
                (camera_depth - (split - transition_width)) / (2.0 * transition_width)
            );
            transition_active = true;
            break;
        }
    }
    float primary_inverse_depth_span = stable_csm_rcp_safe(
        cascade_depth_span[primary],
        0.001
    );
    float secondary_inverse_depth_span = primary_inverse_depth_span;
    if (transition_active) {
        secondary_inverse_depth_span = stable_csm_rcp_safe(
            cascade_depth_span[secondary],
            0.001
        );
    }

    float primary_visibility = stable_csm_pcss_visibility(
        raw_shadow_depth,
        shadow_depth,
        primary,
        shadow_view_proj[primary],
        world_position,
        sample_pixel,
        receiver_normal,
        surface_to_light,
        receiver_ndotl,
        cascade_texel_world[primary],
        cascade_depth_span[primary],
        primary_inverse_depth_span,
        cascade_resolution[primary],
        pcss_params,
        receiver_params,
        debug_filter_radius
    );
    float visibility = primary_visibility;
    if (transition_active) {
        float secondary_visibility = stable_csm_pcss_visibility(
            raw_shadow_depth,
            shadow_depth,
            secondary,
            shadow_view_proj[secondary],
            world_position,
            sample_pixel,
            receiver_normal,
            surface_to_light,
            receiver_ndotl,
            cascade_texel_world[secondary],
            cascade_depth_span[secondary],
            secondary_inverse_depth_span,
            cascade_resolution[secondary],
            pcss_params,
            receiver_params,
            debug_filter_radius
        );
        visibility = lerp(primary_visibility, secondary_visibility, transition_t);
    }

    if (debug_filter_radius) {
        opaque_visibility = visibility;
        transmittance = float3(1.0);
        return float3(visibility);
    }

    uint translucent_bits = uint(max(translucent_mask, 0.0) + 0.5);
    if (translucent_bits == 0u) {
        opaque_visibility = saturate(visibility);
        transmittance = float3(1.0);
        return float3(opaque_visibility);
    }
    float3 primary_transmittance = stable_csm_sample_transmittance(
        primary,
        translucent_shadow_0,
        translucent_shadow_1,
        translucent_shadow_2,
        translucent_shadow_3,
        shadow_view_proj,
        world_position,
        receiver_normal,
        surface_to_light,
        receiver_ndotl,
        cascade_texel_world,
        primary_inverse_depth_span,
        translucent_bits,
        receiver_params.z
    );
    transmittance = primary_transmittance;
    if (transition_active) {
        float3 secondary_transmittance = stable_csm_sample_transmittance(
            secondary,
            translucent_shadow_0,
            translucent_shadow_1,
            translucent_shadow_2,
            translucent_shadow_3,
            shadow_view_proj,
            world_position,
            receiver_normal,
            surface_to_light,
            receiver_ndotl,
            cascade_texel_world,
            secondary_inverse_depth_span,
            translucent_bits,
            receiver_params.z
        );
        transmittance = lerp(primary_transmittance, secondary_transmittance, transition_t);
    }
    opaque_visibility = saturate(visibility);
    return transmittance * opaque_visibility;
}
