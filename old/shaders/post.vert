#version 450

layout(location = 0) out vec2 post_uv;

void main() {
    vec2 position = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    post_uv = position;
    gl_Position = vec4(position * 2.0 - 1.0, 0.0, 1.0);
}
