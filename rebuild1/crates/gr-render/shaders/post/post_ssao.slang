#pragma once

float screen_space_ao(float2 uv, SurfaceMaterial material) {
    if (surface_material_is_background(material)) {
        return 1.0;
    }

    float3 position = view_position(uv, material.source_depth);
    float view_z = max(-position.z, params.depth.x);

    float radius = max(params.ssao.y, 0.001);
    float radius_sq = radius * radius;
    float inner_radius_sq = radius_sq * 0.0324;
    float bias = max(params.ssao.z, max(0.0005, view_z * 0.0015));

    float radius_uv = clamp(
        radius * rcp_safe(view_z, 0.0001) * 0.55,
        0.0012,
        0.038
    );

    int sample_count = int(clamp(params.ssao.w, 1.0, 8.0));
    float inv_sample_count = rcp_safe(float(sample_count), 1.0);

    const float2 kernel[8] = float2[8](
        float2( 0.5381,  0.1856),
        float2(-0.4319,  0.3141),
        float2( 0.2486, -0.7242),
        float2(-0.7198, -0.1937),
        float2( 0.9103,  0.4125),
        float2(-0.3627, -0.9184),
        float2( 0.1269,  0.9872),
        float2(-0.9715,  0.0524)
    );

    float occlusion = 0.0;
    float total = 0.0;

    for (int i = 0; i < sample_count; i++) {

        float scale = lerp(0.28, 1.0, (float(i) + 1.0) * inv_sample_count);
        float2 sample_uv = uv + kernel[i] * radius_uv * scale;

        if (
            any(sample_uv < float2(0.0)) ||
            any(sample_uv > float2(1.0))
        ) {
            continue;
        }

        float sample_depth = depth_at(sample_uv);

        if (is_background_depth(sample_depth)) {
            continue;
        }

        float sample_z = linear_depth(sample_depth);
        float2 sample_ndc = sample_uv * 2.0 - 1.0;
        float3 delta = float3(
            sample_ndc.x * sample_z * params.camera.z - position.x,
            sample_ndc.y * sample_z * params.camera.w - position.y,
            view_z - sample_z
        );
        float distance_sq = dot(delta, delta);

        if (distance_sq <= 0.00000001) {
            continue;
        }

        total += 1.0;
        float closer = step(sample_z + bias, view_z);
        if (closer <= 0.0) {
            continue;
        }

        float range = 1.0 - smoothstep(inner_radius_sq, radius_sq, distance_sq);
        if (range <= 0.0) {
            continue;
        }

        float facing = saturate(dot(material.normal, delta) * rsqrt(distance_sq));

        occlusion += range * facing;
    }

    if (total <= 0.0) {
        return 1.0;
    }

    return clamp(1.0 - occlusion * rcp_safe(total, 1.0), 0.0, 1.0);
}
