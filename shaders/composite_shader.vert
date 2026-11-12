#version 330 core

layout(location = 0) in vec2 a_position;
out vec2 v_texcoord;
out float rad;
out float ang;
out vec3 hue_shader;

uniform float u_time;

void main() {
    v_texcoord = a_position * 0.5 + 0.5;
    rad = length(v_texcoord - 0.5) * 2.0;
    ang = atan(v_texcoord.y - 0.5, v_texcoord.x - 0.5);
    float h = fract(ang / 6.2832 + rad * 0.5 + u_time * 0.05);
    hue_shader = 0.5 + 0.5 * cos(6.2832 * (h + vec3(0.0, 0.33, 0.67)));
    gl_Position = vec4(a_position, 0.0, 1.0);
}
