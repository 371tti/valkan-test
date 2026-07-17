#pragma once

// =========================================================
// Clean Moment Shadow Mapping sampler
//
// Intended input:
//   shadow_cascade_* = RGBA moment map
//   R = z
//   G = z^2 + derivative variance
//   B = z^3
//   A = z^4
//
// This version removes:
//   - PCSS blocker search
//   - Poisson/random rotation
//   - hard/soft visibility mixing
//   - weak-shadow threshold hacks
//
// It keeps:
//   - receiver-plane bias
//   - prefiltered MSM moment sampling
//   - raw moment blocker-distance estimation
//   - 4-moment Hamburger MSM visibility
//   - translucent shadow support
//   - cascade blending
// =========================================================

static const float SHADOW_GAUSSIAN_3[3] = float[3](
    0.27901,
    0.44198,
    0.27901
);

static const float SHADOW_MIN_VARIANCE = 0.000010;
static const float SHADOW_MAX_VARIANCE = 0.000750;
static const float SHADOW_CONTACT_EPSILON = 0.000030;

float shadow_rcp_safe(float value, float floor_value) {
    return 1.0 / max(value, floor_value);
}

float squared(float value) {
    return value * value;
}

float shadow_texel_depth(float texel_world, float depth_span) {
    return texel_world * shadow_rcp_safe(depth_span, 0.001);
}

int2 shadow_texel_coord(Sampler2D moment_map, float2 uv) {
    int2 shadow_size = texture_dimensions(moment_map, 0);
    float2 clamped_uv = clamp(uv, float2(0.0), float2(1.0));

    return clamp(
        int2(clamped_uv * float2(shadow_size)),
        int2(0),
        shadow_size - int2(1)
    );
}

float4 fetch_shadow_moments(Sampler2D moment_map, float2 uv) {
    return load_texel(
        moment_map,
        shadow_texel_coord(moment_map, uv),
        0
    );
}

bool shadow_uv_is_inside(float2 uv) {
    return !(
        uv.x < 0.0 || uv.x > 1.0 ||
        uv.y < 0.0 || uv.y > 1.0
    );
}

float2 shadow_uv(float4 shadow_pos, out float3 projected) {
    projected = shadow_pos.xyz * shadow_rcp_safe(shadow_pos.w, 0.0001);
    return projected.xy * 0.5 + 0.5;
}

bool shadow_projection_is_valid(float3 projected, float2 uv) {
    return !(
        uv.x < 0.0 || uv.x > 1.0 ||
        uv.y < 0.0 || uv.y > 1.0 ||
        projected.z < 0.0 || projected.z > 1.0
    );
}

float reduce_light_bleeding(float visibility, float amount) {
    return clamp(
        (visibility - amount) * shadow_rcp_safe(1.0 - amount, 0.0001),
        0.0,
        1.0
    );
}

// ---------------------------------------------------------
// Bias
// ---------------------------------------------------------

float receiver_plane_shadow_bias(
    float ndotl,
    float3 projected,
    float texel_world,
    float depth_span
) {
    float slope = max(abs(ddx(projected.z)), abs(ddy(projected.z)));
    float grazing = 1.0 - clamp(ndotl, 0.0, 1.0);
    float texel_depth = texel_world * shadow_rcp_safe(depth_span, 0.001);

    return clamp(
        0.000040 +
        grazing * 0.00022 +
        slope * 0.75 +
        texel_depth * lerp(0.16, 0.52, grazing),
        0.000040,
        0.0016
    );
}

float moment_min_variance(
    float ndotl,
    float texel_world,
    float depth_span
) {
    float normalized_texel = shadow_texel_depth(texel_world, depth_span);
    float footprint_variance = squared(normalized_texel * 1.05);
    float grazing_variance = (1.0 - clamp(ndotl, 0.0, 1.0)) * 0.000012;

    return clamp(
        SHADOW_MIN_VARIANCE + footprint_variance + grazing_variance,
        SHADOW_MIN_VARIANCE,
        SHADOW_MAX_VARIANCE
    );
}

// Two moments are enough for a stable, monotonic visibility estimate. Unlike
// the previous four-moment solve this cannot create discontinuous roots when a
// sloped receiver crosses shadow texels.
float variance_shadow_visibility(
    float4 raw_moments,
    float compare,
    float min_variance
) {
    float mean = clamp(raw_moments.x, 0.0, 1.0);
    if (compare <= mean + SHADOW_CONTACT_EPSILON) {
        return 1.0;
    }
    float second = max(raw_moments.y, mean * mean + min_variance);
    float variance = clamp(second - mean * mean, min_variance, SHADOW_MAX_VARIANCE);
    float distance = compare - mean;
    float visibility = variance * shadow_rcp_safe(variance + distance * distance, 0.000001);
    return reduce_light_bleeding(visibility, 0.12);
}

float contact_shadow_visibility(
    Sampler2D raw_shadow_map,
    float4 shadow_pos,
    float ndotl,
    float texel_world,
    float depth_span,
    float4 contact_shadow,
    float filtered_visibility
) {
    float intensity = clamp(contact_shadow.x, 0.0, 1.0);

    if (intensity <= 0.0) {
        return 1.0;
    }

    if (shadow_pos.w <= 0.0 || ndotl <= 0.16) {
        return 1.0;
    }

    float3 projected;
    float2 uv = shadow_uv(shadow_pos, projected);

    if (!shadow_projection_is_valid(projected, uv)) {
        return 1.0;
    }

    float bias = receiver_plane_shadow_bias(
        ndotl,
        projected,
        texel_world,
        depth_span
    );
    float compare = clamp(projected.z - bias, 0.0, 1.0);
    float quality = clamp((contact_shadow.w - 1.0) * 0.04347826, 0.0, 1.0);
    float compare_slope = max(abs(ddx(compare)), abs(ddy(compare)));
    float texel_world_floor = max(texel_world * 0.35, 0.0005);
    float thickness = max(contact_shadow.z, texel_world_floor * 0.12);
    float max_distance = max(
        contact_shadow.y,
        thickness + texel_world_floor * 1.5
    );
    float slope_world = compare_slope * max(depth_span, 0.001);
    float min_separation = max(
        thickness * 0.06,
        texel_world_floor * 0.04 + slope_world * 0.18
    );
    float contact_ramp = max(
        thickness * 0.40,
        texel_world_floor * 0.70
    );

    float4 center_moments = sample_texture(raw_shadow_map, uv);
    float separation =
        (compare - center_moments.x) *
        max(depth_span, 0.001);

    if (separation <= min_separation || separation >= max_distance) {
        return 1.0;
    }

    float visibility_gate = smoothstep(0.04, 0.35, filtered_visibility);
    float near_fade = smoothstep(
        min_separation,
        min_separation + contact_ramp,
        separation
    );
    float slope_stability =
        1.0 - smoothstep(
            thickness,
            max(max_distance * 0.55, thickness + texel_world_floor),
            slope_world
        );
    float light_facing = smoothstep(0.22, 0.60, ndotl);
    float distance_fade = 1.0 - smoothstep(
        thickness,
        max_distance,
        separation
    );
    float contact =
        near_fade *
        distance_fade *
        slope_stability *
        light_facing *
        visibility_gate;
    float darkening = contact *
        intensity *
        lerp(0.72, 1.08, quality);

    return clamp(
        1.0 - darkening,
        max(1.0 - intensity * 0.82, 0.0),
        1.0
    );
}

// ---------------------------------------------------------
// Opaque shadow
// ---------------------------------------------------------

float opaque_shadow_factor(
    Sampler2D filtered_shadow_map,
    float4 shadow_pos,
    float ndotl,
    float texel_world,
    float depth_span
) {
    if (shadow_pos.w <= 0.0) {
        return 1.0;
    }

    float3 projected;
    float2 uv = shadow_uv(shadow_pos, projected);

    if (!shadow_projection_is_valid(projected, uv)) {
        return 1.0;
    }

    float bias = receiver_plane_shadow_bias(
        ndotl,
        projected,
        texel_world,
        depth_span
    );

    float compare = clamp(projected.z - bias, 0.0, 1.0);

    float min_variance = moment_min_variance(
        ndotl,
        texel_world,
        depth_span
    );
    float4 filtered_moments = sample_texture(filtered_shadow_map, uv);
    return variance_shadow_visibility(
        filtered_moments,
        compare,
        min_variance
    );
}

// ---------------------------------------------------------
// Translucent shadow
// ---------------------------------------------------------

float translucent_shadow_depth_bias(float3 projected) {
    float slope = max(abs(ddx(projected.z)), abs(ddy(projected.z)));

    return clamp(
        0.0025 + slope * 4.0,
        0.0025,
        0.020
    );
}

float3 translucent_shadow_rgb(
    Sampler2D transmittance_map,
    float2 uv
) {
    int2 shadow_size = texture_dimensions(transmittance_map, 0);

    float2 texel = float2(
        shadow_rcp_safe(float(shadow_size.x), 1.0),
        shadow_rcp_safe(float(shadow_size.y), 1.0)
    );

    float3 sum = float3(0.0);
    float weight_sum = 0.0;

    const float radius = 0.85;

    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            float weight =
                SHADOW_GAUSSIAN_3[x + 1] *
                SHADOW_GAUSSIAN_3[y + 1];

            float2 sample_uv =
                uv +
                float2(float(x), float(y)) *
                texel *
                radius;

            if (!shadow_uv_is_inside(sample_uv)) {
                continue;
            }

            sum += sample_texture(transmittance_map, sample_uv).rgb * weight;
            weight_sum += weight;
        }
    }

    return clamp(
        sum * shadow_rcp_safe(weight_sum, 0.00001),
        float3(0.0),
        float3(1.0)
    );
}

float3 translucent_shadow_factor(
    Sampler2D transmittance_map,
    float4 shadow_pos
) {
    if (shadow_pos.w <= 0.0) {
        return float3(1.0);
    }

    float3 projected;
    float2 uv = shadow_uv(shadow_pos, projected);

    if (!shadow_projection_is_valid(projected, uv)) {
        return float3(1.0);
    }

    float4 center = sample_texture(transmittance_map, uv);

    float bias = translucent_shadow_depth_bias(projected);

    float receiver_is_behind_translucent_caster =
        step(center.a + bias, projected.z);

    if (receiver_is_behind_translucent_caster <= 0.0) {
        return float3(1.0);
    }

    float3 transmittance = translucent_shadow_rgb(
        transmittance_map,
        uv
    );

    return transmittance;
}

float3 combine_shadow_layers(float3 opaque, float3 translucent) {
    return clamp(
        opaque * translucent,
        float3(0.0),
        float3(1.0)
    );
}

float cascade_transition_width(float split) {
    return max(split * 0.070, 1.80);
}

float2 cascade_transition_bounds(float split) {
    float width = cascade_transition_width(split);

    return float2(
        max(split - width, 0.0),
        split + width
    );
}

float3 cascade_shadow_factor(
    Sampler2D filtered_opaque_map,
    Sampler2D translucent_map,
    float4 shadow_pos,
    float ndotl,
    float texel_world,
    float depth_span,
    float translucent_enabled
) {
    float3 opaque = float3(
        opaque_shadow_factor(
            filtered_opaque_map,
            shadow_pos,
            ndotl,
            texel_world,
            depth_span
        )
    );

    if (translucent_enabled <= 0.5) {
        return opaque;
    }

    float3 translucent = translucent_shadow_factor(
        translucent_map,
        shadow_pos
    );

    return combine_shadow_layers(
        opaque,
        translucent
    );
}

float4 cascade_shadow_position(
    int index,
    float4x4 shadow_view_proj[4],
    float3 world_pos,
    float3 receiver_normal,
    float4 cascade_texel_world,
    float ndotl
) {
    float texel_world = index == 0
        ? cascade_texel_world.x
        : (index == 1 ? cascade_texel_world.y : (index == 2 ? cascade_texel_world.z : cascade_texel_world.w));
    float grazing = 1.0 - clamp(ndotl, 0.0, 1.0);
    float normal_offset = max(texel_world, 0.0) * lerp(0.30, 1.10, grazing);
    float4 position = float4(world_pos + receiver_normal * normal_offset, 1.0);
    if (index == 0) {
        return mul(shadow_view_proj[0], position);
    }
    if (index == 1) {
        return mul(shadow_view_proj[1], position);
    }
    if (index == 2) {
        return mul(shadow_view_proj[2], position);
    }
    return mul(shadow_view_proj[3], position);
}

float3 sample_shadow_cascade(
    int index,
    Sampler2D shadow_cascade_0,
    Sampler2D shadow_cascade_1,
    Sampler2D shadow_cascade_2,
    Sampler2D shadow_cascade_3,
    Sampler2D raw_shadow_cascade_0,
    Sampler2D raw_shadow_cascade_1,
    Sampler2D raw_shadow_cascade_2,
    Sampler2D raw_shadow_cascade_3,
    Sampler2D translucent_shadow_0,
    Sampler2D translucent_shadow_1,
    Sampler2D translucent_shadow_2,
    Sampler2D translucent_shadow_3,
    float4 shadow_pos,
    float4 cascade_texel_world,
    float4 cascade_depth_span,
    float translucent_enabled,
    float ndotl
) {
    if (index == 0) {
        return cascade_shadow_factor(
            shadow_cascade_0,
            translucent_shadow_0,
            shadow_pos,
            ndotl,
            cascade_texel_world.x,
            cascade_depth_span.x,
            translucent_enabled
        );
    }

    if (index == 1) {
        return cascade_shadow_factor(
            shadow_cascade_1,
            translucent_shadow_1,
            shadow_pos,
            ndotl,
            cascade_texel_world.y,
            cascade_depth_span.y,
            translucent_enabled
        );
    }

    if (index == 2) {
        return cascade_shadow_factor(
            shadow_cascade_2,
            translucent_shadow_2,
            shadow_pos,
            ndotl,
            cascade_texel_world.z,
            cascade_depth_span.z,
            translucent_enabled
        );
    }

    return cascade_shadow_factor(
        shadow_cascade_3,
        translucent_shadow_3,
        shadow_pos,
        ndotl,
        cascade_texel_world.w,
        cascade_depth_span.w,
        translucent_enabled
    );
}

float sample_contact_shadow_cascade(
    int index,
    Sampler2D raw_shadow_cascade_0,
    Sampler2D raw_shadow_cascade_1,
    Sampler2D raw_shadow_cascade_2,
    Sampler2D raw_shadow_cascade_3,
    float4 shadow_pos,
    float4 cascade_texel_world,
    float4 cascade_depth_span,
    float4 contact_shadow,
    float ndotl,
    float filtered_visibility
) {
    if (index == 0) {
        return contact_shadow_visibility(
            raw_shadow_cascade_0,
            shadow_pos,
            ndotl,
            cascade_texel_world.x,
            cascade_depth_span.x,
            contact_shadow,
            filtered_visibility
        );
    }

    if (index == 1) {
        return contact_shadow_visibility(
            raw_shadow_cascade_1,
            shadow_pos,
            ndotl,
            cascade_texel_world.y,
            cascade_depth_span.y,
            contact_shadow,
            filtered_visibility
        );
    }

    if (index == 2) {
        return contact_shadow_visibility(
            raw_shadow_cascade_2,
            shadow_pos,
            ndotl,
            cascade_texel_world.z,
            cascade_depth_span.z,
            contact_shadow,
            filtered_visibility
        );
    }

    return contact_shadow_visibility(
        raw_shadow_cascade_3,
        shadow_pos,
        ndotl,
        cascade_texel_world.w,
        cascade_depth_span.w,
        contact_shadow,
        filtered_visibility
    );
}

float shadow_max_visibility(float3 shadow) {
    return max(max(shadow.r, shadow.g), shadow.b);
}

float shadow_lit_contact_support(float3 shadow) {
    return smoothstep(0.72, 0.96, shadow_max_visibility(shadow));
}

float4 compose_shadow_result(float3 shadow, float contact_visibility, bool contact_enabled) {
    if (!contact_enabled) {
        return float4(shadow, 1.0);
    }

    float contact_support = shadow_lit_contact_support(shadow);

    return float4(
        shadow,
        lerp(1.0, contact_visibility, contact_support)
    );
}

float4 shadow_factor(
    Sampler2D shadow_cascade_0,
    Sampler2D shadow_cascade_1,
    Sampler2D shadow_cascade_2,
    Sampler2D shadow_cascade_3,
    Sampler2D raw_shadow_cascade_0,
    Sampler2D raw_shadow_cascade_1,
    Sampler2D raw_shadow_cascade_2,
    Sampler2D raw_shadow_cascade_3,
    Sampler2D translucent_shadow_0,
    Sampler2D translucent_shadow_1,
    Sampler2D translucent_shadow_2,
    Sampler2D translucent_shadow_3,
    float4x4 shadow_view_proj[4],
    float3 world_pos,
    float3 receiver_normal,
    float camera_depth,
    float4 cascade_splits,
    float4 cascade_texel_world,
    float4 cascade_depth_span,
    float4 contact_shadow,
    float translucent_enabled,
    float ndotl
) {
    float fade_width = max(cascade_splits.w * 0.07, 1.8);
    if (camera_depth >= cascade_splits.w + fade_width) {
        return float4(1.0);
    }

    int cascade_index =
        int(camera_depth >= cascade_splits.x) +
        int(camera_depth >= cascade_splits.y) +
        int(camera_depth >= cascade_splits.z);
    float4 shadow_pos = cascade_shadow_position(
        cascade_index,
        shadow_view_proj,
        world_pos,
        receiver_normal,
        cascade_texel_world,
        ndotl
    );
    float3 shadow = sample_shadow_cascade(
        cascade_index,
        shadow_cascade_0,
        shadow_cascade_1,
        shadow_cascade_2,
        shadow_cascade_3,
        raw_shadow_cascade_0,
        raw_shadow_cascade_1,
        raw_shadow_cascade_2,
        raw_shadow_cascade_3,
        translucent_shadow_0,
        translucent_shadow_1,
        translucent_shadow_2,
        translucent_shadow_3,
        shadow_pos,
        cascade_texel_world,
        cascade_depth_span,
        translucent_enabled,
        ndotl
    );

    bool contact_enabled = contact_shadow.x > 0.0;
    float contact_visibility = 1.0;
    if (contact_enabled && shadow_lit_contact_support(shadow) > 0.0) {
        contact_visibility = sample_contact_shadow_cascade(
            cascade_index,
            raw_shadow_cascade_0,
            raw_shadow_cascade_1,
            raw_shadow_cascade_2,
            raw_shadow_cascade_3,
            shadow_pos,
            cascade_texel_world,
            cascade_depth_span,
            contact_shadow,
            ndotl,
            shadow_max_visibility(shadow)
        );
    }

    float4 result = compose_shadow_result(shadow, contact_visibility, contact_enabled);
    float distance_fade = smoothstep(
        cascade_splits.w - fade_width,
        cascade_splits.w + fade_width,
        camera_depth
    );
    return lerp(result, float4(1.0), distance_fade);
}

