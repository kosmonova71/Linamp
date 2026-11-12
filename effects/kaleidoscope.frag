#version 330 core
in vec2 v_texcoord;
out vec4 FragColor;

uniform sampler2D u_source;
uniform vec2 u_resolution;
uniform int u_segments;
uniform float u_angle;
uniform float u_time;
uniform float u_cx;
uniform float u_cy;

void main() {
    vec2 center = vec2(u_cx, u_cy);
    vec2 uv = v_texcoord - center;

    float segments = float(u_segments);
    float segment_angle = 2.0 * 3.14159 / segments;

    float rotation = u_angle + u_time * 0.1;

    float r = length(uv);
    float theta = atan(uv.y, uv.x);

    theta = mod(theta - rotation, segment_angle);
    if (theta > segment_angle * 0.5) {
        theta = segment_angle - theta;
    }

    vec2 kaleido_uv = center + vec2(cos(theta + rotation), sin(theta + rotation)) * r;
    kaleido_uv = fract(kaleido_uv);

    FragColor = texture(u_source, kaleido_uv);
}
