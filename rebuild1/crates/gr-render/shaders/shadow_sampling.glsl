#ifndef REBUILD1_SHADOW_SAMPLING_GLSL
#define REBUILD1_SHADOW_SAMPLING_GLSL

const int SHADOW_MAX_PCF_TAPS = 49;

const float SHADOW_GAUSSIAN_5[5] = float[5](
    0.06136,
    0.24477,
    0.38774,
    0.24477,
    0.06136
);

const float SHADOW_GAUSSIAN_7[7] = float[7](
    0.03663,
    0.11128,
    0.21675,
    0.27068,
    0.21675,
    0.11128,
    0.03663
);

const float SHADOW_PCSS_LIGHT_RADIUS_WORLD = 0.42;
const float SHADOW_PCSS_MIN_BLOCKER_DEPTH = 0.025;
const float SHADOW_PCSS_MAX_SEARCH_RADIUS = 22.0;
const float SHADOW_PCSS_MAX_FILTER_RADIUS = 16.0;

float shadow_rcp_safe(float value, float floor_value) {
    return 1.0 / max(value, floor_value);
}

float shadow_visibility_at(
    sampler2D shadow_map,
    vec2 uv,
    float compare,
    vec2 offset,
    float depth_softness
) {
    float sampled_depth = texture(shadow_map, uv + offset).r;
    return smoothstep(-depth_softness, depth_softness, sampled_depth - compare);
}

int shadow_filter_tap_count(int base_count, float radius) {
    int count = min(max(base_count, 25), SHADOW_MAX_PCF_TAPS);
    if (radius > 4.0) {
        count = max(count, 49);
    }
    return count;
}

float shadow_filter_radius(float ndotl) {
    float grazing = 1.0 - clamp(ndotl, 0.0, 1.0);
    return mix(1.15, 2.25, grazing);
}

void shadow_find_average_blocker(
    sampler2D shadow_map,
    vec2 uv,
    float compare,
    vec2 texel,
    float search_radius,
    out float average_depth,
    out float blocker_count
) {
    float depth_sum = 0.0;
    blocker_count = 0.0;

    for (int y = -2; y <= 2; y++) {
        for (int x = -2; x <= 2; x++) {
            vec2 offset = vec2(float(x), float(y)) * 0.5;
            float weight =
                SHADOW_GAUSSIAN_5[x + 2] *
                SHADOW_GAUSSIAN_5[y + 2];
            float sampled_depth = texture(
                shadow_map,
                uv + offset * texel * search_radius
            ).r;
            float gap = compare - sampled_depth;

            if (gap > 0.0) {
                depth_sum += sampled_depth * weight;
                blocker_count += weight;
            }
        }
    }

    if (blocker_count <= 0.0) {
        average_depth = compare;
        return;
    }

    average_depth = depth_sum / blocker_count;
}

float gaussian_shadow_visibility(
    sampler2D shadow_map,
    vec2 uv,
    float compare,
    vec2 texel,
    float radius,
    float depth_softness,
    int tap_count
) {
    bool use_large_kernel = tap_count >= 49;
    float kernel_scale = use_large_kernel ? 0.33333334 : 0.5;
    float sum = 0.0;

    for (int y = -3; y <= 3; y++) {
        for (int x = -3; x <= 3; x++) {
            if (!use_large_kernel && (abs(x) > 2 || abs(y) > 2)) {
                continue;
            }

            float weight;
            if (use_large_kernel) {
                weight = SHADOW_GAUSSIAN_7[x + 3] * SHADOW_GAUSSIAN_7[y + 3];
            } else {
                weight = SHADOW_GAUSSIAN_5[x + 2] * SHADOW_GAUSSIAN_5[y + 2];
            }
            vec2 offset = vec2(float(x), float(y)) * kernel_scale;
            sum += shadow_visibility_at(
                shadow_map,
                uv,
                compare,
                offset * texel * radius,
                depth_softness
            ) * weight;
        }
    }

    return clamp(sum, 0.0, 1.0);
}

float pcss_lite_filter_radius(
    sampler2D shadow_map,
    vec2 uv,
    float compare,
    vec2 texel,
    float base_radius,
    float texel_world,
    float depth_span,
    float ndotl
) {
    float grazing = 1.0 - clamp(ndotl, 0.0, 1.0);
    float inv_texel_world = shadow_rcp_safe(texel_world, 0.001);
    float light_radius_texels = clamp(
        SHADOW_PCSS_LIGHT_RADIUS_WORLD * inv_texel_world,
        1.5,
        SHADOW_PCSS_MAX_SEARCH_RADIUS
    );
    float search_radius = clamp(
        base_radius + light_radius_texels * mix(0.75, 1.35, grazing),
        base_radius + 1.0,
        SHADOW_PCSS_MAX_SEARCH_RADIUS
    );

    float average_blocker_depth;
    float blocker_count;
    shadow_find_average_blocker(
        shadow_map,
        uv,
        compare,
        texel,
        search_radius,
        average_blocker_depth,
        blocker_count
    );

    if (blocker_count <= 0.0) {
        return base_radius;
    }

    float blocker_gap = max(compare - average_blocker_depth, 0.0);
    float blocker_gap_world = blocker_gap * max(depth_span, 1.0);
    float perspective_penumbra =
        blocker_gap *
        shadow_rcp_safe(average_blocker_depth, SHADOW_PCSS_MIN_BLOCKER_DEPTH) *
        light_radius_texels *
        0.82;
    float world_penumbra =
        blocker_gap_world *
        inv_texel_world *
        0.026;
    float penumbra = max(perspective_penumbra, world_penumbra);
    float filter_limit = clamp(
        light_radius_texels * mix(0.58, 1.08, grazing),
        4.0,
        SHADOW_PCSS_MAX_FILTER_RADIUS
    );

    return base_radius + clamp(penumbra, 0.0, filter_limit);
}

float receiver_plane_shadow_bias(float ndotl, vec3 projected) {
    float slope = max(abs(dFdx(projected.z)), abs(dFdy(projected.z)));
    float grazing = 1.0 - clamp(ndotl, 0.0, 1.0);

    return clamp(
        0.00035 + grazing * 0.0018 + slope * 2.0,
        0.00035,
        0.006
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

float opaque_shadow_factor(
    sampler2D shadow_map,
    vec4 shadow_pos,
    float ndotl,
    float texel_world,
    float depth_span,
    int tap_count
) {
    if (shadow_pos.w <= 0.0) {
        return 1.0;
    }

    vec3 projected;
    vec2 uv = shadow_uv(shadow_pos, projected);

    if (!shadow_projection_is_valid(projected, uv)) {
        return 1.0;
    }

    ivec2 shadow_size = textureSize(shadow_map, 0);
    vec2 texel = vec2(
        shadow_rcp_safe(float(shadow_size.x), 1.0),
        shadow_rcp_safe(float(shadow_size.y), 1.0)
    );

    float compare = projected.z - receiver_plane_shadow_bias(ndotl, projected);
    float base_radius = shadow_filter_radius(ndotl);
    int base_count = clamp(tap_count, 1, SHADOW_MAX_PCF_TAPS);
    float radius = pcss_lite_filter_radius(
        shadow_map,
        uv,
        compare,
        texel,
        base_radius,
        texel_world,
        depth_span,
        ndotl
    );
    int count = shadow_filter_tap_count(base_count, radius);

    float depth_softness = clamp(
        radius * texel_world * shadow_rcp_safe(depth_span, 1.0) * 0.34,
        0.00035,
        0.00380
    );

    return gaussian_shadow_visibility(
        shadow_map,
        uv,
        compare,
        texel,
        radius,
        depth_softness,
        count
    );
}

float translucent_shadow_depth_bias(vec3 projected) {
    float slope = max(abs(dFdx(projected.z)), abs(dFdy(projected.z)));

    return clamp(
        0.0030 + slope * 5.0,
        0.0030,
        0.025
    );
}

vec4 gaussian_translucent_shadow_sample(sampler2D transmittance_map, vec2 uv) {
    ivec2 shadow_size = textureSize(transmittance_map, 0);
    vec2 texel = vec2(
        shadow_rcp_safe(float(shadow_size.x), 1.0),
        shadow_rcp_safe(float(shadow_size.y), 1.0)
    );
    vec4 sum = vec4(0.0);
    const float radius = 1.35;

    for (int y = -2; y <= 2; y++) {
        for (int x = -2; x <= 2; x++) {
            float weight =
                SHADOW_GAUSSIAN_5[x + 2] *
                SHADOW_GAUSSIAN_5[y + 2];
            vec2 offset = vec2(float(x), float(y)) * texel * radius;
            sum += texture(transmittance_map, uv + offset) * weight;
        }
    }

    return clamp(sum, vec4(0.0), vec4(1.0));
}

vec3 translucent_shadow_factor(sampler2D transmittance_map, vec4 shadow_pos) {
    if (shadow_pos.w <= 0.0) {
        return vec3(1.0);
    }

    vec3 projected;
    vec2 uv = shadow_uv(shadow_pos, projected);

    if (!shadow_projection_is_valid(projected, uv)) {
        return vec3(1.0);
    }

    vec4 transmittance = gaussian_translucent_shadow_sample(transmittance_map, uv);

    float bias = translucent_shadow_depth_bias(projected);

    float receiver_is_behind_translucent_caster =
        step(transmittance.a + bias, projected.z);

    return mix(
        vec3(1.0),
        transmittance.rgb,
        receiver_is_behind_translucent_caster
    );
}

vec3 combine_shadow_layers(vec3 opaque, vec3 translucent) {
    float visible = max(max(opaque.r, opaque.g), opaque.b);
    float tint_weight = smoothstep(0.20, 0.85, visible);
    return mix(opaque, opaque * translucent, tint_weight);
}

float cascade_transition_width(float split) {
    return max(split * 0.10, 3.0);
}

float squared(float value) {
    return value * value;
}

vec3 cascade_shadow_factor(
    sampler2D opaque_map,
    sampler2D translucent_map,
    vec4 shadow_pos,
    float ndotl,
    float texel_world,
    float depth_span,
    int tap_count,
    float translucent_enabled
) {
    vec3 opaque = vec3(
        opaque_shadow_factor(
            opaque_map,
            shadow_pos,
            ndotl,
            texel_world,
            depth_span,
            tap_count
        )
    );

    if (translucent_enabled <= 0.5) {
        return opaque;
    }

    vec3 translucent = translucent_shadow_factor(translucent_map, shadow_pos);
    return combine_shadow_layers(opaque, translucent);
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
    float near_width = cascade_transition_width(cascade_splits.x);
    float near_min = cascade_splits.x - near_width;
    float near_max = cascade_splits.x + near_width;

    if (camera_distance_sq < squared(near_min)) {
        return cascade_shadow_factor(
            shadow_cascade_0,
            translucent_shadow_0,
            shadow_pos[0],
            ndotl,
            cascade_texel_world.x,
            cascade_depth_span.x,
            15,
            translucent_enabled
        );
    }

    if (camera_distance_sq <= squared(near_max)) {
        float t = smoothstep(
            squared(near_min),
            squared(near_max),
            camera_distance_sq
        );

        vec3 near_shadow = cascade_shadow_factor(
            shadow_cascade_0,
            translucent_shadow_0,
            shadow_pos[0],
            ndotl,
            cascade_texel_world.x,
            cascade_depth_span.x,
            15,
            translucent_enabled
        );

        vec3 mid_shadow = cascade_shadow_factor(
            shadow_cascade_1,
            translucent_shadow_1,
            shadow_pos[1],
            ndotl,
            cascade_texel_world.y,
            cascade_depth_span.y,
            13,
            translucent_enabled
        );

        return mix(near_shadow, mid_shadow, t);
    }

    float mid_width = cascade_transition_width(cascade_splits.y);
    float mid_min = cascade_splits.y - mid_width;
    float mid_max = cascade_splits.y + mid_width;

    if (camera_distance_sq < squared(mid_min)) {
        return cascade_shadow_factor(
            shadow_cascade_1,
            translucent_shadow_1,
            shadow_pos[1],
            ndotl,
            cascade_texel_world.y,
            cascade_depth_span.y,
            13,
            translucent_enabled
        );
    }

    if (camera_distance_sq <= squared(mid_max)) {
        float t = smoothstep(
            squared(mid_min),
            squared(mid_max),
            camera_distance_sq
        );

        vec3 mid_shadow = cascade_shadow_factor(
            shadow_cascade_1,
            translucent_shadow_1,
            shadow_pos[1],
            ndotl,
            cascade_texel_world.y,
            cascade_depth_span.y,
            13,
            translucent_enabled
        );

        vec3 mid_far_shadow = cascade_shadow_factor(
            shadow_cascade_2,
            translucent_shadow_2,
            shadow_pos[2],
            ndotl,
            cascade_texel_world.z,
            cascade_depth_span.z,
            11,
            translucent_enabled
        );

        return mix(mid_shadow, mid_far_shadow, t);
    }

    float far_width = cascade_transition_width(cascade_splits.z);
    float far_min = cascade_splits.z - far_width;
    float far_max = cascade_splits.z + far_width;

    if (camera_distance_sq < squared(far_min)) {
        return cascade_shadow_factor(
            shadow_cascade_2,
            translucent_shadow_2,
            shadow_pos[2],
            ndotl,
            cascade_texel_world.z,
            cascade_depth_span.z,
            11,
            translucent_enabled
        );
    }

    if (camera_distance_sq <= squared(far_max)) {
        float t = smoothstep(
            squared(far_min),
            squared(far_max),
            camera_distance_sq
        );

        vec3 mid_far_shadow = cascade_shadow_factor(
            shadow_cascade_2,
            translucent_shadow_2,
            shadow_pos[2],
            ndotl,
            cascade_texel_world.z,
            cascade_depth_span.z,
            11,
            translucent_enabled
        );

        vec3 far_shadow = cascade_shadow_factor(
            shadow_cascade_3,
            translucent_shadow_3,
            shadow_pos[3],
            ndotl,
            cascade_texel_world.w,
            cascade_depth_span.w,
            9,
            translucent_enabled
        );

        return mix(mid_far_shadow, far_shadow, t);
    }

    return cascade_shadow_factor(
        shadow_cascade_3,
        translucent_shadow_3,
        shadow_pos[3],
        ndotl,
        cascade_texel_world.w,
        cascade_depth_span.w,
        9,
        translucent_enabled
    );
}

#endif
