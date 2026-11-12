#version 330 core
in vec2 v_texcoord;
out vec4 FragColor;

uniform float u_time;
uniform vec4 u_audio;
uniform vec2 u_resolution;
uniform sampler2D u_prev_frame;
uniform float u_feedback;
uniform vec2 u_zoom;
uniform float u_rot;
uniform float u_warp;
uniform float u_decay;
uniform float u_beat_flash;
uniform sampler2D u_texture;
uniform float u_texture_blend;
uniform float u_texture_zoom;
uniform vec2 u_texture_offset;
uniform float u_texture_distort;
uniform float u_brightness;

// Improved noise functions
float noise(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

float smoothNoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);

    float a = noise(i);
    float b = noise(i + vec2(1.0, 0.0));
    float c = noise(i + vec2(0.0, 1.0));
    float d = noise(i + vec2(1.0, 1.0));

    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm(vec2 p) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;

    for (int i = 0; i < 6; i++) {
        value += amplitude * smoothNoise(p * frequency);
        frequency *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}

// 3D noise function for more complex patterns
float noise3D(vec3 p) {
    vec2 q = vec2(fbm(p.xy), fbm(p.yz));
    return fbm(p.xy + q);
}

// HSV to RGB conversion
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// Plasma effect function
vec3 plasma(vec2 p, float time) {
    float v1 = sin(p.x * 10.0 + time);
    float v2 = sin(10.0 * (p.x * sin(time / 2.0) + p.y * cos(time / 3.0)) + time);
    float v3 = sin(sqrt(100.0 * (p.x * p.x + p.y * p.y) + 1.0) + time);
    float v = v1 + v2 + v3;
    return vec3(sin(v), sin(v + 2.0), sin(v + 4.0)) * 0.5 + 0.5;
}

void main() {
    vec2 uv = v_texcoord;
    vec2 center = vec2(0.5, 0.5);

    // Apply zoom and rotation
    vec2 p = (uv - center) * u_zoom.x;
    float angle = u_rot;
    mat2 rot = mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
    p = rot * p + center;

    // Enhanced warp effects with multiple layers
    float warp_strength = max(u_warp * 8.0, 0.5) * (1.0 + u_audio.x * 1.2 + u_audio.w * 0.8);

    // Multi-frequency warp for complexity
    p.x += sin(p.y * 20.0 + u_time * 4.0) * 0.03 * warp_strength;
    p.y += cos(p.x * 18.0 + u_time * 3.5) * 0.03 * warp_strength;
    p += sin(u_time * 5.0 + p * 40.0) * 0.02 * warp_strength;

    // Beat-reactive vortex
    if (u_audio.w > 0.5) {
        float beat_intensity = (u_audio.w - 0.5) * 2.0;
        float beat_angle = u_time * 15.0 * beat_intensity;
        mat2 beat_rot = mat2(cos(beat_angle), -sin(beat_angle), sin(beat_angle), cos(beat_angle));
        p = beat_rot * (p - center) + center;
    }

    // Complex pattern generation with 3D noise
    vec3 p3d = vec3(p * 6.0, u_time * 0.5);
    float n1 = noise3D(p3d);
    p3d = vec3(p * 12.0, u_time * -0.3);
    float n2 = noise3D(p3d);
    p3d = vec3(p * 24.0, u_time * 0.7);
    float n3 = noise3D(p3d);

    float pattern = n1 * 0.4 + n2 * 0.3 + n3 * 0.2 + fbm(p * 32.0) * 0.1;

    // QUANTUM FIELD EFFECT
    vec2 quantum_d = uv - center;
    float quantum_r = length(quantum_d);
    float quantum_field = sin(quantum_r * 30.0 - u_time * 8.0) * cos(quantum_r * 25.0 + u_time * 6.0);
    quantum_field *= exp(-quantum_r * 2.0) * warp_strength;
    p += quantum_d * quantum_field * 0.1;

    // DIMENSIONAL RIFT EFFECT
    float rift_angle = atan(quantum_d.y, quantum_d.x);
    float rift_intensity = sin(rift_angle * 8.0 + u_time * 3.0) * cos(rift_angle * 5.0 - u_time * 2.0);
    rift_intensity *= (1.0 + u_audio.x * 1.5) * warp_strength;
    p.x += sin(uv.y * rift_angle * 20.0) * rift_intensity * 0.02;
    p.y += cos(uv.x * rift_angle * 20.0) * rift_intensity * 0.02;

    // NEBULA CLOUD EFFECT
    vec2 nebula_p = p * 15.0;
    float nebula = fbm(nebula_p + u_time * 0.3);
    nebula = pow(nebula, 2.0) * (1.0 + u_audio.y * 0.8);

    // Regenerate pattern with all distortions
    n1 = fbm(p * 8.0 + u_time * 1.0);
    n2 = fbm(p * 16.0 - u_time * 0.7);
    n3 = fbm(p * 32.0 + u_time * 1.5);
    pattern = n1 * 0.35 + n2 * 0.25 + n3 * 0.15 + nebula * 0.15 + quantum_field * 0.1;

    // Enhanced color dynamics
    float hue = fract(u_time * 0.2 + length(p - center) * 1.2 + u_audio.y * 0.6 + u_audio.x * 0.4);
    hue = mix(hue, fract(hue + 0.33), u_audio.w);  // Beat shifts hue
    float saturation = 0.7 + 0.3 * sin(u_time * 1.5 + u_audio.z * 3.0);
    saturation = mix(saturation, 1.0, u_audio.w * 0.5);  // Beats increase saturation
    float value = 0.3 + pattern * 0.6 + u_brightness * 0.3;

    vec3 color = hsv2rgb(vec3(hue, saturation, value));

    // Enhanced audio-reactive effects
    color += vec3(u_audio.x * 0.6, u_audio.y * 0.4, u_audio.z * 0.3);
    color += u_audio.w * 0.4;  // Beat flash

    // ENERGY MATRIX OVERLAY
    vec2 matrix = fract(p * 25.0 + u_time * 0.8);
    float matrix_lines = step(matrix.x, 0.03) * step(matrix.y, 0.03);
    vec3 matrix_color = vec3(0.0, 1.0, 0.5) * matrix_lines * 0.3;
    matrix_color *= (1.0 + u_audio.z * 2.0);  // Treble drives matrix
    color += matrix_color;

    // PLASMA INTEGRATION
    vec3 plasma_color = plasma(p, u_time * 2.0);
    color = mix(color, plasma_color, 0.2 + u_audio.w * 0.3);

    // PARTICLE FIELD EFFECT
    float particle_density = fbm(p * 50.0 + u_time * 2.0);
    float particles = step(particle_density, 0.98);
    vec3 particle_color = vec3(1.0, 0.8, 0.2) * particles * 0.4;
    particle_color *= (1.0 + u_audio.x * 1.5);
    color += particle_color;

    // GLOW AND BLOOM
    float glow = smoothstep(0.4, 1.0, pattern);
    vec3 glow_color = hsv2rgb(vec3(fract(hue + 0.5), 0.8, glow));
    color += glow_color * 0.3;

    // Feedback integration
    vec2 feedback_uv = p;
    feedback_uv = fract(feedback_uv);
    vec4 feedback_color = texture(u_prev_frame, feedback_uv);
    color = mix(color, feedback_color.rgb, u_feedback * 0.6);

    // Texture blending
    if (u_texture_blend > 0.0) {
        vec2 tex_uv = p * vec2(u_texture_zoom) + u_texture_offset;
        tex_uv = fract(tex_uv);
        vec4 tex_color = texture(u_texture, tex_uv);
        color = mix(color, tex_color.rgb, u_texture_blend);
    }

    // Final enhancements
    color = pow(color, vec3(0.8));  // Gamma correction
    color = clamp(color, 0.0, 1.0);

    FragColor = vec4(color, 1.0);
}
