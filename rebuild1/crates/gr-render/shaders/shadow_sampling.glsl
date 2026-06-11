#ifndef REBUILD1_SHADOW_SAMPLING_GLSL
#define REBUILD1_SHADOW_SAMPLING_GLSL

const vec2 SHADOW_PCF_OFFSETS[16] = vec2[16](
    vec2(0.0, 0.0),
    vec2(1.0, 0.0),
    vec2(-1.0, 0.0),
    vec2(0.0, 1.0),
    vec2(0.0, -1.0),
    vec2(0.75, 0.75),
    vec2(-0.75, 0.75),
    vec2(0.75, -0.75),
    vec2(-0.75, -0.75),
    vec2(1.7, 0.55),
    vec2(-1.7, 0.55),
    vec2(1.7, -0.55),
    vec2(-1.7, -0.55),
    vec2(0.55, 1.7),
    vec2(-0.55, 1.7),
    vec2(0.55, -1.7)
);

float shadow_rcp_safe(float value, float floor_value) {
    return 1.0 / max(value, floor_value);
}

float shadow_visibility_at(sampler2D shadow_map, vec2 uv, float compare, vec2 offset) {
    return step(compare, texture(shadow_map, uv + offset).r);
}

float shadow_filter_radius(float ndotl) {
    float grazing = 1.0 - clamp(ndotl, 0.0, 1.0);
    return mix(1.5, 3.0, grazing);
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
    float radius = shadow_filter_radius(ndotl);

    float sum = 0.0;
    int count = clamp(tap_count, 1, 16);

    for (int i = 0; i < count; i++) {
        sum += shadow_visibility_at(
            shadow_map,
            uv,
            compare,
            SHADOW_PCF_OFFSETS[i] * texel * radius
        );
    }

    // 元の softened / inversesqrt は削除。
    // 影の丸めは PCF 平均だけにする。
    return clamp(sum / float(count), 0.0, 1.0);
}

float translucent_shadow_depth_bias(vec3 projected) {
    float slope = max(abs(dFdx(projected.z)), abs(dFdy(projected.z)));

    return clamp(
        0.0030 + slope * 5.0,
        0.0030,
        0.025
    );
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

    vec4 transmittance = clamp(
        texture(transmittance_map, uv),
        vec4(0.0),
        vec4(1.0)
    );

    float bias = translucent_shadow_depth_bias(projected);

    float receiver_is_behind_translucent_caster =
        step(transmittance.a + bias, projected.z);

    return mix(
        vec3(1.0),
        transmittance.rgb,
        receiver_is_behind_translucent_caster
    );
}

float cascade_transition_width(float split) {
    // 元: max(split * 0.08, 2.0)
    // 遷移範囲を狭くして、2カスケード同時サンプリング領域を減らす。
    return max(split * 0.04, 1.0);
}

float squared(float value) {
    return value * value;
}

vec3 cascade_shadow_factor(
    sampler2D opaque_map,
    sampler2D translucent_map,
    vec4 shadow_pos,
    float ndotl,
    int tap_count,
    float translucent_enabled
) {
    vec3 opaque = vec3(
        opaque_shadow_factor(
            opaque_map,
            shadow_pos,
            ndotl,
            tap_count
        )
    );

    if (translucent_enabled <= 0.5) {
        return opaque;
    }

    return opaque * translucent_shadow_factor(
        translucent_map,
        shadow_pos
    );
}

vec3 shadow_factor(
    sampler2D shadow_cascade_0,
    sampler2D shadow_cascade_1,
    sampler2D shadow_cascade_2,
    sampler2D translucent_shadow_0,
    sampler2D translucent_shadow_1,
    sampler2D translucent_shadow_2,
    vec4 shadow_pos[3],
    float camera_distance_sq,
    vec4 cascade_splits,
    float ndotl
) {
    float translucent_enabled = cascade_splits.w;

    float near_width = cascade_transition_width(cascade_splits.x);
    float near_min = cascade_splits.x - near_width;
    float near_max = cascade_splits.x + near_width;

    if (camera_distance_sq < squared(near_min)) {
        return cascade_shadow_factor(
            shadow_cascade_0,
            translucent_shadow_0,
            shadow_pos[0],
            ndotl,
            12, // 元: 16
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
            12, // 元: 16
            translucent_enabled
        );

        vec3 mid_shadow = cascade_shadow_factor(
            shadow_cascade_1,
            translucent_shadow_1,
            shadow_pos[1],
            ndotl,
            6, // 元: 10
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
            6, // 元: 10
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
            6, // 元: 10
            translucent_enabled
        );

        vec3 far_shadow = cascade_shadow_factor(
            shadow_cascade_2,
            translucent_shadow_2,
            shadow_pos[2],
            ndotl,
            4, // 元: 6
            translucent_enabled
        );

        return mix(mid_shadow, far_shadow, t);
    }

    return cascade_shadow_factor(
        shadow_cascade_2,
        translucent_shadow_2,
        shadow_pos[2],
        ndotl,
        4, // 元: 6
        translucent_enabled
    );
}

#endif