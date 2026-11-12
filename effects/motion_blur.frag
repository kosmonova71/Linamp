#version 330 core
in vec2 v_texcoord;
out vec4 FragColor;

uniform sampler2D u_source;
uniform sampler2D u_previous_frame;
uniform vec2 u_resolution;
uniform float u_strength;
uniform int u_samples;
uniform float u_time;

void main() {
    vec2 texel_size = 1.0 / u_resolution;
    vec4 color = vec4(0.0);
    float total_weight = 0.0;

    float blur_strength = u_strength;

    vec2 motion_vector = vec2(
        sin(u_time * 0.45) * 0.4,
        cos(u_time * 0.28) * 0.4
    ) * blur_strength * 0.75;

    int samples = clamp(u_samples, 1, 16);
    for (int i = 0; i < samples; i++) {
        float t = float(i) / max(float(samples - 1), 1.0);
        vec2 offset = motion_vector * (t - 0.5);
        float weight = 1.0 - abs(t - 0.5) * 0.8;
        weight = max(weight, 0.1);

        color += texture(u_source, v_texcoord + offset) * weight;
        total_weight += weight;
    }

    color /= max(total_weight, 0.00001);
    FragColor = color;
}
