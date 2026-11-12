#version 330 core
in vec2 v_texcoord;
out vec4 FragColor;

uniform sampler2D u_source;
uniform vec2 u_resolution;
uniform float u_intensity;
uniform float u_threshold;
uniform float u_time;

float random(vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
}

void main() {
    vec2 uv = v_texcoord;

    float trigger = smoothstep(u_threshold - 0.05, u_threshold + 0.05, u_intensity);
    float glitch_amount = u_intensity * trigger;

    if (glitch_amount > 0.02) {
        float shift = random(vec2(u_time * 12.0, uv.y * 37.0)) * glitch_amount * 0.12;
        vec4 color;
        color.r = texture(u_source, uv + vec2(shift, 0.0)).r;
        color.g = texture(u_source, uv).g;
        color.b = texture(u_source, uv - vec2(shift, 0.0)).b;
        color.a = texture(u_source, uv).a;

        float block_size = 0.01 + glitch_amount * 0.03;
        vec2 block_uv = floor(uv / block_size) * block_size;
        float block_noise = random(block_uv + u_time * 1.3);

        if (block_noise < glitch_amount * 0.6) {
            float swap = random(block_uv);
            if (swap < 0.33) {
                color.rb = color.br;
            } else if (swap < 0.66) {
                color.rg = color.gr;
            }
        }

        float scanline = sin(uv.y * 1200.0 + u_time * 12.0) * 0.5 + 0.5;
        color *= mix(1.0, scanline, glitch_amount * 0.25);

        FragColor = color;
    } else {
        FragColor = texture(u_source, uv);
    }
}
