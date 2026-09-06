#pragma once

// This is intentionally a separate include from post_god_ray_common.slang. Only the mask pass
// owns the camera UBO and directional shadow-map descriptor; the legacy radial/prefilter passes
// keep their smaller descriptor layouts.
struct GodRayFrameCamera {
    float4x4 view_proj;
    float4x4 view;
    float4x4 shadow_view_proj[4];
    float4 stable_csm_pcss_params;
    float4 stable_csm_receiver_params;
    float4 shadow_cascade_splits;
    float4 shadow_cascade_texel_world;
    float4 shadow_cascade_depth_span;
    float4 shadow_cascade_resolution;
    float4 camera_pos;
    float4 light_dir;
    float4 light_color;
    float4 ambient_color;
};

[[vk::binding(0, 1)]] ConstantBuffer<GodRayFrameCamera> god_ray_frame_camera;
[[vk::binding(3, 0)]] Sampler2DArrayShadow god_ray_shadow_depth;
// Deep translucent CSM targets: RGB stores additive log-transmittance and A stores the nearest
// transparent depth. They use the mask set so the dedicated camera-ray march can apply the same
// participating-medium attenuation as the surface and dedicated camera-ray paths.
[[vk::binding(6, 0)]] Sampler2D god_ray_translucent_shadow_0;
[[vk::binding(7, 0)]] Sampler2D god_ray_translucent_shadow_1;
[[vk::binding(8, 0)]] Sampler2D god_ray_translucent_shadow_2;
[[vk::binding(9, 0)]] Sampler2D god_ray_translucent_shadow_3;

static const float GOD_RAY_PI = 3.14159265;

float god_ray_phase(float cosine) {
    // A broad Henyey-Greenstein lobe keeps the medium visible away from the sun while still
    // producing a directional shaft when the camera looks toward the light.
    const float anisotropy = 0.28;
    float g2 = anisotropy * anisotropy;
    float denominator = pow(
        max(1.0 + g2 - 2.0 * anisotropy * cosine, 0.001),
        1.5
    );
    float hg = (1.0 - g2) / (4.0 * GOD_RAY_PI * denominator);
    return hg * 0.72 + 0.030;
}

uint god_ray_cascade_for_depth(float view_depth) {
    uint cascade = 0u;
    cascade += view_depth > god_ray_frame_camera.shadow_cascade_splits.x ? 1u : 0u;
    cascade += view_depth > god_ray_frame_camera.shadow_cascade_splits.y ? 1u : 0u;
    cascade += view_depth > god_ray_frame_camera.shadow_cascade_splits.z ? 1u : 0u;
    return min(cascade, 3u);
}

float god_ray_shadow_visibility_for_cascade(
    float3 world_position,
    uint cascade,
    float view_depth,
    out bool projection_valid
) {
    projection_valid = true;
    if (god_ray_frame_camera.light_dir.w <= 0.5) {
        return 1.0;
    }

    float4 shadow_position = mul(
        god_ray_frame_camera.shadow_view_proj[cascade],
        float4(world_position, 1.0)
    );
    if (shadow_position.w <= 0.0001) {
        projection_valid = false;
        return 1.0;
    }

    float3 projected = shadow_position.xyz / shadow_position.w;
    float2 uv = projected.xy * 0.5 + 0.5;
    if (
        any(uv <= float2(0.001)) ||
        any(uv >= float2(0.999)) ||
        projected.z <= 0.0 ||
        projected.z >= 1.0
    ) {
        projection_valid = false;
        return 1.0;
    }

    float resolution = max(god_ray_frame_camera.shadow_cascade_resolution[cascade], 1.0);
    float texel_depth = god_ray_frame_camera.shadow_cascade_texel_world[cascade] *
        rcp_safe(god_ray_frame_camera.shadow_cascade_depth_span[cascade], 0.001);
    float bias = max(
        4.0 / 65535.0,
        0.28 * texel_depth + god_ray_frame_camera.stable_csm_receiver_params.x * 0.00025
    );
    float2 texel = float2(rcp_safe(resolution, 1.0));
    float visibility = 0.0;
    visibility += god_ray_shadow_depth.SampleCmpLevelZero(
        float3(uv + texel * float2(-0.45, -0.45), float(cascade)),
        projected.z - bias
    );
    visibility += god_ray_shadow_depth.SampleCmpLevelZero(
        float3(uv + texel * float2( 0.45, -0.45), float(cascade)),
        projected.z - bias
    );
    visibility += god_ray_shadow_depth.SampleCmpLevelZero(
        float3(uv + texel * float2(-0.45,  0.45), float(cascade)),
        projected.z - bias
    );
    visibility += god_ray_shadow_depth.SampleCmpLevelZero(
        float3(uv + texel * float2( 0.45,  0.45), float(cascade)),
        projected.z - bias
    );

    return visibility * 0.25;
}

float god_ray_shadow_visibility(float3 world_position, float view_depth) {
    if (god_ray_frame_camera.light_dir.w <= 0.5) {
        return 1.0;
    }

    uint primary = god_ray_cascade_for_depth(view_depth);
    uint secondary = primary;
    float transition_t = 0.0;

    // Match the CPU Stable CSM overlap (4.5% of the split, with a 1-unit floor). The volume
    // marches through camera depth, so using the same pair/blend as the surface shader prevents
    // a visible brightness step when a shaft crosses a cascade boundary.
    for (uint boundary = 0u; boundary < 3u; boundary++) {
        float split = god_ray_frame_camera.shadow_cascade_splits[boundary];
        float transition_width = max(split * 0.10, 1.0);
        if (abs(view_depth - split) < transition_width) {
            primary = boundary;
            secondary = boundary + 1u;
            transition_t = saturate(
                (view_depth - (split - transition_width)) /
                (2.0 * transition_width)
            );
            break;
        }
    }

    bool primary_valid;
    float primary_visibility = god_ray_shadow_visibility_for_cascade(
        world_position,
        primary,
        view_depth,
        primary_valid
    );
    if (secondary == primary) {
        // A projection outside the selected map is not a lit sample. Returning the helper's
        // fail-open value here used to paint the invalid CSM rectangle into the shaft whenever
        // the camera-ray march crossed a map edge.
        return primary_valid ? primary_visibility : 1.0;
    }

    bool secondary_valid;
    float secondary_visibility = god_ray_shadow_visibility_for_cascade(
        world_position,
        secondary,
        view_depth,
        secondary_valid
    );
    // During overlap, only blend projections that actually cover the sample. The old code
    // blended a fail-open `1.0` from an invalid projection with a valid shadow result, producing
    // a camera-angle-dependent bright slab at the map boundary.
    if (primary_valid && secondary_valid) {
        return lerp(primary_visibility, secondary_visibility, transition_t);
    }
    if (primary_valid) {
        return primary_visibility;
    }
    if (secondary_valid) {
        return secondary_visibility;
    }
    return 1.0;
}

float3 god_ray_translucent_sample(
    Sampler2D shadow_map,
    uint cascade,
    float3 world_position,
    out bool projection_valid
) {
    projection_valid = true;
    float4 shadow_position = mul(
        god_ray_frame_camera.shadow_view_proj[cascade],
        float4(world_position, 1.0)
    );
    if (shadow_position.w <= 0.0001) {
        projection_valid = false;
        return float3(1.0);
    }

    float3 projected = shadow_position.xyz / shadow_position.w;
    float2 uv = projected.xy * 0.5 + 0.5;
    if (
        any(uv <= float2(0.001)) ||
        any(uv >= float2(0.999)) ||
        projected.z <= 0.0 ||
        projected.z >= 1.0
    ) {
        projection_valid = false;
        return float3(1.0);
    }

    float resolution = max(god_ray_frame_camera.shadow_cascade_resolution[cascade], 1.0);
    float texel = rcp_safe(resolution, 1.0);
    float depth_tolerance = max(
        god_ray_frame_camera.shadow_cascade_texel_world[cascade] *
            rcp_safe(god_ray_frame_camera.shadow_cascade_depth_span[cascade], 0.001),
        2.0 / 65535.0
    ) * 2.0;
    float2 offsets[4] = {
        float2(-0.5, -0.5),
        float2( 0.5, -0.5),
        float2(-0.5,  0.5),
        float2( 0.5,  0.5)
    };
    float3 transmittance = float3(0.0);
    [unroll]
    for (uint tap = 0u; tap < 4u; ++tap) {
        float2 sample_uv = clamp(uv + offsets[tap] * texel, texel, float2(1.0) - texel);
        float4 value = shadow_map.SampleLevel(sample_uv, 0.0);
        float behind = smoothstep(
            -depth_tolerance,
            depth_tolerance,
            projected.z + depth_tolerance - value.a
        );
        float3 layer_transmittance = exp(clamp(value.rgb, float3(-16.0), float3(0.0)));
        transmittance += lerp(float3(1.0), layer_transmittance, behind);
    }
    return transmittance * 0.25;
}

float3 god_ray_translucent_transmittance(float3 world_position, float view_depth) {
    uint mask = uint(max(god_ray_frame_camera.light_color.w, 0.0) + 0.5);
    if (mask == 0u) {
        return float3(1.0);
    }

    // The far cascade is the continuous medium projection used by the dedicated volume. Prefer
    // it whenever available so transparent silhouettes cannot introduce camera-depth layers.
    if ((mask & (1u << 3u)) != 0u) {
        bool valid;
        float3 far_transmittance = god_ray_translucent_sample(
            god_ray_translucent_shadow_3,
            3u,
            world_position,
            valid
        );
        if (valid) {
            return far_transmittance;
        }
    }

    uint cascade = god_ray_cascade_for_depth(view_depth);
    if (cascade == 0u && (mask & 1u) != 0u) {
        bool valid;
        float3 value = god_ray_translucent_sample(
            god_ray_translucent_shadow_0,
            0u,
            world_position,
            valid
        );
        if (valid) {
            return value;
        }
    }
    if (cascade <= 1u && (mask & (1u << 1u)) != 0u) {
        bool valid;
        float3 value = god_ray_translucent_sample(
            god_ray_translucent_shadow_1,
            1u,
            world_position,
            valid
        );
        if (valid) {
            return value;
        }
    }
    if ((mask & (1u << 2u)) != 0u) {
        bool valid;
        float3 value = god_ray_translucent_sample(
            god_ray_translucent_shadow_2,
            2u,
            world_position,
            valid
        );
        if (valid) {
            return value;
        }
    }
    return float3(1.0);
}

float3 god_ray_world_ray(float2 uv) {
    float aspect = max(params.target.z, 0.0001);
    float focal = max(params.target.w, 0.0001);
    // `target.z` is height / width, matching the post camera coefficients. Keep the
    // reconstructed ray identical to the scene projection: X uses f * height / width and Y
    // uses -f. The inverse-aspect pair stretches shafts at widescreen resolutions.
    float focal_x = focal * aspect;
    float focal_y = -focal;
    float2 ndc = uv * 2.0 - 1.0;
    float3 view_ray = normalize_fast(float3(ndc.x / focal_x, ndc.y / focal_y, -1.0));

    // The push lanes carry the camera's orthonormal basis only for the volumetric mode. This
    // avoids adding a second camera uniform while keeping the legacy source payload unchanged.
    float3 right = params.source0.xyz;
    float3 up = params.color0.xyz;
    float3 forward = params.source1.xyz;
    return normalize_fast(right * view_ray.x + up * view_ray.y - forward * view_ray.z);
}

struct GodRayVolumeResult {
    float3 scattering;
    float transmittance;
};

float god_ray_fraction(float value) {
    return value - floor(value);
}

float god_ray_blue_noise(float2 uv) {
    // Interleaved-gradient noise is a compact analytic blue-noise tile. It has no texture
    // descriptor or cache miss, and the frame phase is a Cranley-Patterson rotation so the
    // sequence moves every frame instead of repeating the same depth layers.
    float2 texel = max(params.target.xy, float2(0.000001));
    float2 pixel = floor(uv / texel);
    float frame = params.source1.w;
    float gradient = god_ray_fraction(
        pixel.x * 0.06711056 +
        pixel.y * 0.00583715 +
        frame * 0.754877666
    );
    float interleaved = god_ray_fraction(52.9829189 * gradient);
    return god_ray_fraction(interleaved + frame * 0.618033989);
}

// This is the dedicated Sun Shaft transport path. It runs at half resolution, so 48 camera-ray
// samples keep the cost close to the old quarter-resolution 32-step path while reducing the
// physical interval between shadow evaluations enough that a light-space silhouette cannot be
// copied into visible depth layers.
static const int GOD_RAY_VOLUME_SAMPLES = 48;
static const float GOD_RAY_FOG_DENSITY = 0.0035;
static const float GOD_RAY_FOG_HEIGHT = 0.0;
static const float GOD_RAY_FOG_HEIGHT_FALLOFF = 0.018;

float god_ray_medium_density(float3 world_position) {
    if (params.features.w <= 0.5) {
        // Keep a small scattering medium for the high God Ray profile even when the caller has
        // disabled the broader inspection fog profile.
        return GOD_RAY_FOG_DENSITY * 0.42;
    }

    float height_delta = world_position.y - GOD_RAY_FOG_HEIGHT;
    float height_factor = exp(-height_delta * GOD_RAY_FOG_HEIGHT_FALLOFF);
    return GOD_RAY_FOG_DENSITY * clamp(height_factor, 0.06, 3.0);
}

GodRayVolumeResult god_ray_volumetric_integral(float2 uv, float scene_depth) {
    GodRayVolumeResult result;
    result.scattering = float3(0.0);
    result.transmittance = 1.0;
    if (params.bloom.w <= 0.0 && params.features.w <= 0.5) {
        return result;
    }

    float near_distance = max(params.depth.x, 0.001);
    // `params.depth.y` is the compact raster camera far plane and can be much shorter than the
    // configured participating medium. The CPU puts the independent medium range in source0.w;
    // using the raster far here would recreate an angle-dependent cutoff whenever the scene clip
    // range is optimized around a small visible model.
    float max_distance = max(params.source0.w, near_distance + 0.05);
    float end_distance = god_ray_background_depth(scene_depth)
        ? max_distance
        : min(god_ray_linear_depth(scene_depth), max_distance);
    end_distance = max(end_distance, near_distance + 0.01);
    float3 camera_ray = god_ray_world_ray(uv);
    // `scene_depth` and the medium limits are camera-forward view depth. Convert them to distance
    // travelled along the oblique world ray before constructing samples and Beer-Lambert steps.
    // Treating view depth as Euclidean ray distance is angle-dependent and was the remaining
    // source of the side-on slab pattern after shadow-map filtering changes.
    float3 camera_forward = normalize_fast(params.source1.xyz);
    float ray_cos = max(dot(camera_ray, camera_forward), 0.25);
    float step_length = (end_distance - near_distance) /
        (float(GOD_RAY_VOLUME_SAMPLES) * ray_cos);
    float cosine = dot(-camera_ray, god_ray_frame_camera.light_dir.xyz);
    float phase = god_ray_phase(cosine);
    float3 sun_radiance = max(god_ray_frame_camera.light_color.rgb, float3(0.0));
    float3 ambient_radiance = max(god_ray_frame_camera.ambient_color.rgb, float3(0.0));
    float noise = god_ray_blue_noise(uv);
    float3 scattering = float3(0.0);
    float transmittance = 1.0;

    for (int index = 0; index < GOD_RAY_VOLUME_SAMPLES; index++) {
        // Jitter every stratum, not the entire ray by one common offset. A common offset leaves
        // a coherent staircase whenever the blocker silhouette is close to a sample boundary;
        // a per-stratum Cranley-Patterson rotation gives temporal integration independent samples.
        float sample_noise = god_ray_fraction(noise + float(index) * 0.618033989);
        float t = (float(index) + 0.5 + (sample_noise - 0.5) * 0.78) /
            float(GOD_RAY_VOLUME_SAMPLES);
        t = saturate(t);
        float view_depth = near_distance + (end_distance - near_distance) * t;
        float3 world_position = god_ray_frame_camera.camera_pos.xyz +
            camera_ray * (view_depth / ray_cos);
        float density = god_ray_medium_density(world_position);
        float segment_optical_depth = density * step_length;
        float segment_transmittance = exp(-segment_optical_depth);
        float visibility = god_ray_shadow_visibility(world_position, view_depth);
        float3 translucent = god_ray_translucent_transmittance(world_position, view_depth);
        float3 direct = sun_radiance * phase * visibility * translucent *
            clamp(params.bloom.w, 0.0, 1.0);
        float3 ambient = ambient_radiance * 0.42 * (params.features.w > 0.5 ? 1.0 : 0.0);
        scattering += (direct + ambient) * density * step_length * transmittance;
        transmittance *= segment_transmittance;
    }

    result.scattering = min(max(scattering, float3(0.0)), float3(2.0));
    result.transmittance = saturate(transmittance);
    return result;
}
