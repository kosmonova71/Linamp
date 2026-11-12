#version 330 core
in vec2 v_texcoord;
out vec4 FragColor;

uniform sampler2D u_source;
uniform vec2 u_resolution;
uniform float u_strength;
uniform float u_time;
uniform float u_cx;
uniform float u_cy;

void main() {
    vec2 center = vec2(u_cx, u_cy);
    vec2 uv = v_texcoord - center;

    float distortion = u_strength;

    float r = length(uv);
    float theta = atan(uv.y, uv.x);

    float distortion_amount = distortion * (1.0 + sin(u_time * 2.0) * 0.15);
    r *= 1.0 + distortion_amount * r * r * 0.6;

    vec2 distorted_uv = center + vec2(cos(theta), sin(theta)) * r;
    distorted_uv = clamp(distorted_uv, 0.001, 0.999);

    float aberration = clamp(distortion_amount * 0.02, 0.0, 0.04);
    vec2 red_uv = clamp(distorted_uv + vec2(aberration, 0.0), 0.0, 1.0);
    vec2 green_uv = distorted_uv;
    vec2 blue_uv = clamp(distorted_uv - vec2(aberration, 0.0), 0.0, 1.0);

    vec4 color;
    color.r = texture(u_source, red_uv).r;
    color.g = texture(u_source, green_uv).g;
    color.b = texture(u_source, blue_uv).b;
    color.a = 1.0;
    FragColor = color;
}
