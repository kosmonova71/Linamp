#version 330 core

in vec2 v_texcoord;
in float rad;
in float ang;
in vec3 hue_shader;

out vec4 FragColor;

uniform sampler2D u_texture;
uniform float u_time;
uniform vec4 u_audio;
uniform float u_brightness;

void main() {
    vec3 color = texture(u_texture, v_texcoord).rgb;

    float hue_mix = 0.05 + 0.1 * (u_audio.x + u_audio.y + u_audio.z);
    color = mix(color, color * hue_shader, hue_mix);

    color *= u_brightness;

    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 1.2));

    color = clamp(color, 0.0, 1.0);
    FragColor = vec4(color, 1.0);
}
