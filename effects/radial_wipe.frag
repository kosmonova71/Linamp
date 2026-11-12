#version 330 core
in vec2 v_uv;
out vec4 FragColor;
uniform sampler2D u_from;
uniform sampler2D u_to;
uniform float u_progress;
void main() {
    float angle = (atan(v_uv.y - 0.5, v_uv.x - 0.5) + 3.14159) / (2.0 * 3.14159);
    FragColor = mix(texture(u_from, v_uv), texture(u_to, v_uv), step(angle, u_progress));
}