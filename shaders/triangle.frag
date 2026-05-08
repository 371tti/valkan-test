#version 450

layout(set = 0, binding = 0) uniform Scene {
    mat4 view_proj;
    vec4 light_dir;
    vec4 light_color;
    vec4 ambient;
    vec4 camera_pos;
} scene;

layout(location = 0) in vec3 frag_normal;
layout(location = 1) in vec2 frag_uv;
layout(location = 2) in vec3 frag_world_pos;
layout(location = 3) in vec4 frag_base_color;
layout(location = 0) out vec4 out_color;

void main() {
    vec3 normal = normalize(frag_normal);
    vec3 light = normalize(-scene.light_dir.xyz);
    vec3 view = normalize(scene.camera_pos.xyz - frag_world_pos);
    vec3 half_vec = normalize(light + view);

    float diffuse = max(dot(normal, light), 0.0);
    float specular = pow(max(dot(normal, half_vec), 0.0), 32.0) * 0.22;
    vec3 uv_tint = mix(vec3(0.78, 0.88, 1.08), vec3(1.08, 0.82, 0.74), frag_uv.x);
    vec3 lit = scene.ambient.rgb + scene.light_color.rgb * (diffuse + specular);

    out_color = vec4(frag_base_color.rgb * uv_tint * lit, frag_base_color.a);
}
