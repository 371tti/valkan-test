#ifndef REBUILD1_SHADOW_SAMPLING_GLSL
#define REBUILD1_SHADOW_SAMPLING_GLSL

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
//   - variable penumbra radius
//   - Poisson/random rotation
//   - hard/soft visibility mixing
//   - weak-shadow threshold hacks
//
// It keeps:
//   - receiver-plane bias
//   - prefiltered moment map sampling
//   - 4-moment Hamburger MSM visibility
//   - translucent shadow support
//   - cascade blending
// =========================================================

const float SHADOW_GAUSSIAN_3[3] = float[3](
    0.27901,
    0.44198,
    0.27901
);

const float SHADOW_MSM_MIN_VARIANCE = 0.000010;
const float SHADOW_MSM_MAX_VARIANCE = 0.000750;
const float SHADOW_MSM_LIGHT_BLEED_REDUCTION = 0.22;
const float SHADOW_MSM_MOMENT_BIAS = 0.000020;

const float SHADOW_CONTACT_EPSILON = 0.000030;

float shadow_rcp_safe(float value, float floor_value) {
    return 1.0 / max(value, floor_value);
}

float squared(float value) {
    return value * value;
}

bool shadow_uv_is_inside(vec2 uv) {
    return !(
        uv.x < 0.0 || uv.x > 1.0 ||
        uv.y < 0.0 || uv.y > 1.0
    );
}

vec2 shadow_uv(vec4 shadow_pos, out vec3 projected) {
    projected = shadow_pos.xyz * shadow_rcp_safe(shadow_pos.w, 0.0001);
    return projected.xy * 0.5 + 0.5;
}

bool shadow_projection_is_valid(vec3 projected, vec2 uv) {
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
    vec3 projected,
    float texel_world,
    float depth_span
) {
    float slope = max(abs(dFdx(projected.z)), abs(dFdy(projected.z)));
    float grazing = 1.0 - clamp(ndotl, 0.0, 1.0);
    float texel_depth = texel_world * shadow_rcp_safe(depth_span, 0.001);

    return clamp(
        0.000040 +
        grazing * 0.00022 +
        slope * 0.75 +
        texel_depth * mix(0.16, 0.52, grazing),
        0.000040,
        0.0016
    );
}

float moment_min_variance(
    float ndotl,
    float texel_world,
    float depth_span
) {
    float normalized_texel = texel_world * shadow_rcp_safe(depth_span, 0.001);
    float footprint_variance = squared(normalized_texel * 1.05);
    float grazing_variance = (1.0 - clamp(ndotl, 0.0, 1.0)) * 0.000012;

    return clamp(
        SHADOW_MSM_MIN_VARIANCE + footprint_variance + grazing_variance,
        SHADOW_MSM_MIN_VARIANCE,
        SHADOW_MSM_MAX_VARIANCE
    );
}

// ---------------------------------------------------------
// Moment loading / filtering
// ---------------------------------------------------------

vec4 sanitize_shadow_moments(vec4 moments, float min_variance) {
    moments = clamp(moments, vec4(0.0), vec4(1.0));

    float m1 = moments.x;
    float m2 = max(moments.y, m1 * m1 + min_variance);

    // Do not aggressively "repair" m3/m4.
    // Over-repairing makes blocky soft islands more likely.
    float m3 = clamp(moments.z, 0.0, 1.0);
    float m4 = max(moments.w, m2 * m2 + min_variance * min_variance);

    return vec4(m1, m2, m3, m4);
}

vec4 sample_shadow_moments(
    sampler2D moment_map,
    vec2 uv,
    float min_variance
) {
    return sanitize_shadow_moments(
        texture(moment_map, uv),
        min_variance
    );
}

// ---------------------------------------------------------
// MSM visibility
// ---------------------------------------------------------

float vsm_fallback_visibility(
    vec4 moments,
    float compare,
    float min_variance
) {
    float mean = moments.x;

    if (compare <= mean + SHADOW_CONTACT_EPSILON) {
        return 1.0;
    }

    float second = max(moments.y, mean * mean + min_variance);

    float variance = clamp(
        second - mean * mean,
        min_variance,
        SHADOW_MSM_MAX_VARIANCE
    );

    float distance = compare - mean;

    float chebyshev =
        variance *
        shadow_rcp_safe(
            variance + distance * distance,
            0.000001
        );

    return reduce_light_bleeding(
        chebyshev,
        SHADOW_MSM_LIGHT_BLEED_REDUCTION
    );
}

// Hamburger 4-moment MSM visibility.
// Returns 1.0 for lit and 0.0 for fully shadowed.
float msm_hamburger_visibility(
    vec4 raw_moments,
    float compare,
    float min_variance
) {
    vec4 moments = sanitize_shadow_moments(raw_moments, min_variance);

    moments = mix(
        moments,
        vec4(0.5),
        SHADOW_MSM_MOMENT_BIAS
    );

    float z0 = clamp(compare, 0.0, 1.0);

    if (z0 <= moments.x + SHADOW_CONTACT_EPSILON) {
        return 1.0;
    }

    float l32d22 = moments.z - moments.x * moments.y;
    float d22 = moments.y - moments.x * moments.x;
    float squared_depth_variance = moments.w - moments.y * moments.y;

    d22 = max(d22, min_variance);
    squared_depth_variance = max(squared_depth_variance, min_variance);

    float d33d22 =
        squared_depth_variance * d22 -
        l32d22 * l32d22;

    if (d33d22 <= min_variance * min_variance) {
        return vsm_fallback_visibility(
            moments,
            z0,
            min_variance
        );
    }

    float inv_d22 = shadow_rcp_safe(d22, min_variance);
    float l32 = l32d22 * inv_d22;

    vec3 c = vec3(1.0, z0, z0 * z0);

    c.y -= moments.x;
    c.z -= moments.y + l32 * c.y;

    c.y *= inv_d22;
    c.z *= d22 * shadow_rcp_safe(
        d33d22,
        min_variance * min_variance
    );

    c.y -= l32 * c.z;
    c.x -= dot(c.yz, moments.xy);

    if (abs(c.z) <= 0.000001) {
        return vsm_fallback_visibility(
            moments,
            z0,
            min_variance
        );
    }

    float inv_cz = 1.0 / c.z;
    float p = c.y * inv_cz;
    float q = c.x * inv_cz;

    float discriminant = max(p * p * 0.25 - q, 0.0);
    float root = sqrt(discriminant);

    float z1 = -p * 0.5 - root;
    float z2 = -p * 0.5 + root;

    vec4 switch_value = z2 < z0
        ? vec4(z1, z0, 1.0, 1.0)
        : (
            z1 < z0
                ? vec4(z0, z1, 0.0, 1.0)
                : vec4(0.0)
        );

    float denominator =
        (z2 - switch_value.y) *
        (z0 - z1);

    if (abs(denominator) <= 0.000001) {
        return vsm_fallback_visibility(
            moments,
            z0,
            min_variance
        );
    }

    float quotient =
        (
            switch_value.x * z2 -
            moments.x * (switch_value.x + z2) +
            moments.y
        ) / denominator;

    float shadow_intensity = clamp(
        switch_value.z + switch_value.w * quotient,
        0.0,
        1.0
    );

    float visibility = 1.0 - shadow_intensity;

    // Mild reduction only. Too much reduction causes cutout-like blotches.
    return reduce_light_bleeding(
        visibility,
        SHADOW_MSM_LIGHT_BLEED_REDUCTION * 0.25
    );
}

// ---------------------------------------------------------
// Opaque shadow
// ---------------------------------------------------------

float opaque_shadow_factor(
    sampler2D shadow_map,
    vec4 shadow_pos,
    float ndotl,
    float texel_world,
    float depth_span
) {
    if (shadow_pos.w <= 0.0) {
        return 1.0;
    }

    vec3 projected;
    vec2 uv = shadow_uv(shadow_pos, projected);

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

    vec4 moments = sample_shadow_moments(
        shadow_map,
        uv,
        min_variance
    );

    return msm_hamburger_visibility(
        moments,
        compare,
        min_variance
    );
}

// ---------------------------------------------------------
// Translucent shadow
// ---------------------------------------------------------

float translucent_shadow_depth_bias(vec3 projected) {
    float slope = max(abs(dFdx(projected.z)), abs(dFdy(projected.z)));

    return clamp(
        0.0025 + slope * 4.0,
        0.0025,
        0.020
    );
}

vec3 translucent_shadow_rgb(
    sampler2D transmittance_map,
    vec2 uv
) {
    ivec2 shadow_size = textureSize(transmittance_map, 0);

    vec2 texel = vec2(
        shadow_rcp_safe(float(shadow_size.x), 1.0),
        shadow_rcp_safe(float(shadow_size.y), 1.0)
    );

    vec3 sum = vec3(0.0);
    float weight_sum = 0.0;

    const float radius = 0.85;

    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            float weight =
                SHADOW_GAUSSIAN_3[x + 1] *
                SHADOW_GAUSSIAN_3[y + 1];

            vec2 sample_uv =
                uv +
                vec2(float(x), float(y)) *
                texel *
                radius;

            if (!shadow_uv_is_inside(sample_uv)) {
                continue;
            }

            sum += texture(transmittance_map, sample_uv).rgb * weight;
            weight_sum += weight;
        }
    }

    return clamp(
        sum * shadow_rcp_safe(weight_sum, 0.00001),
        vec3(0.0),
        vec3(1.0)
    );
}

vec3 translucent_shadow_factor(
    sampler2D transmittance_map,
    vec4 shadow_pos
) {
    if (shadow_pos.w <= 0.0) {
        return vec3(1.0);
    }

    vec3 projected;
    vec2 uv = shadow_uv(shadow_pos, projected);

    if (!shadow_projection_is_valid(projected, uv)) {
        return vec3(1.0);
    }

    vec4 center = texture(transmittance_map, uv);

    float bias = translucent_shadow_depth_bias(projected);

    float receiver_is_behind_translucent_caster =
        step(center.a + bias, projected.z);

    vec3 transmittance = translucent_shadow_rgb(
        transmittance_map,
        uv
    );

    return mix(
        vec3(1.0),
        transmittance,
        receiver_is_behind_translucent_caster
    );
}

vec3 combine_shadow_layers(vec3 opaque, vec3 translucent) {
    return clamp(
        opaque * translucent,
        vec3(0.0),
        vec3(1.0)
    );
}

float cascade_transition_width(float split) {
    return max(split * 0.070, 1.80);
}

vec2 cascade_transition_bounds(float split) {
    float width = cascade_transition_width(split);

    return vec2(
        max(split - width, 0.0),
        split + width
    );
}

vec3 cascade_shadow_factor(
    sampler2D opaque_map,
    sampler2D translucent_map,
    vec4 shadow_pos,
    float ndotl,
    float texel_world,
    float depth_span,
    float translucent_enabled
) {
    vec3 opaque = vec3(
        opaque_shadow_factor(
            opaque_map,
            shadow_pos,
            ndotl,
            texel_world,
            depth_span
        )
    );

    if (translucent_enabled <= 0.5) {
        return opaque;
    }

    vec3 translucent = translucent_shadow_factor(
        translucent_map,
        shadow_pos
    );

    return combine_shadow_layers(
        opaque,
        translucent
    );
}

vec3 sample_shadow_cascade(
    int index,
    sampler2D shadow_cascade_0,
    sampler2D shadow_cascade_1,
    sampler2D shadow_cascade_2,
    sampler2D shadow_cascade_3,
    sampler2D translucent_shadow_0,
    sampler2D translucent_shadow_1,
    sampler2D translucent_shadow_2,
    sampler2D translucent_shadow_3,
    vec4 shadow_pos[4],
    vec4 cascade_texel_world,
    vec4 cascade_depth_span,
    float translucent_enabled,
    float ndotl
) {
    if (index == 0) {
        return cascade_shadow_factor(
            shadow_cascade_0,
            translucent_shadow_0,
            shadow_pos[0],
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
            shadow_pos[1],
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
            shadow_pos[2],
            ndotl,
            cascade_texel_world.z,
            cascade_depth_span.z,
            translucent_enabled
        );
    }

    return cascade_shadow_factor(
        shadow_cascade_3,
        translucent_shadow_3,
        shadow_pos[3],
        ndotl,
        cascade_texel_world.w,
        cascade_depth_span.w,
        translucent_enabled
    );
}

vec3 shadow_factor(
    sampler2D shadow_cascade_0,
    sampler2D shadow_cascade_1,
    sampler2D shadow_cascade_2,
    sampler2D shadow_cascade_3,
    sampler2D translucent_shadow_0,
    sampler2D translucent_shadow_1,
    sampler2D translucent_shadow_2,
    sampler2D translucent_shadow_3,
    vec4 shadow_pos[4],
    float camera_distance_sq,
    vec4 cascade_splits,
    vec4 cascade_texel_world,
    vec4 cascade_depth_span,
    float translucent_enabled,
    float ndotl
) {
    float split_depths[4] = float[4](
        cascade_splits.x,
        cascade_splits.y,
        cascade_splits.z,
        cascade_splits.w
    );

    for (int index = 0; index < 3; index++) {
        vec2 transition = cascade_transition_bounds(
            split_depths[index]
        );

        float transition_min_sq = squared(transition.x);
        float transition_max_sq = squared(transition.y);

        if (camera_distance_sq < transition_min_sq) {
            return sample_shadow_cascade(
                index,
                shadow_cascade_0,
                shadow_cascade_1,
                shadow_cascade_2,
                shadow_cascade_3,
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
        }

        if (camera_distance_sq <= transition_max_sq) {
            float t = smoothstep(
                transition_min_sq,
                transition_max_sq,
                camera_distance_sq
            );

            vec3 lower = sample_shadow_cascade(
                index,
                shadow_cascade_0,
                shadow_cascade_1,
                shadow_cascade_2,
                shadow_cascade_3,
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

            vec3 upper = sample_shadow_cascade(
                index + 1,
                shadow_cascade_0,
                shadow_cascade_1,
                shadow_cascade_2,
                shadow_cascade_3,
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

            return mix(lower, upper, t);
        }
    }

    return sample_shadow_cascade(
        3,
        shadow_cascade_0,
        shadow_cascade_1,
        shadow_cascade_2,
        shadow_cascade_3,
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
}

#endif
