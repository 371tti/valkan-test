#ifndef COMMON_GLSL
#define COMMON_GLSL

const float PI = 3.14159265359;

float saturate(float value) {
    return clamp(value, 0.0, 1.0);
}

float luma(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

float square(float value) {
    return value * value;
}

float linear_depth(float depth) {
    float near_plane = max(scene.debug_params.y, 0.0001);
    float far_plane = max(scene.debug_params.z, near_plane + 0.0001);
    float denom = max(far_plane - depth * (far_plane - near_plane), 0.0001);

    return near_plane * far_plane / denom;
}

vec3 environment_color(vec3 direction) {
    return vec3(0.0);
}

vec3 safe_direction(vec3 direction) {
    return vec3(
        abs(direction.x) < 0.0001 ? (direction.x < 0.0 ? -0.0001 : 0.0001) : direction.x,
        abs(direction.y) < 0.0001 ? (direction.y < 0.0 ? -0.0001 : 0.0001) : direction.y,
        abs(direction.z) < 0.0001 ? (direction.z < 0.0 ? -0.0001 : 0.0001) : direction.z
    );
}

#endif
