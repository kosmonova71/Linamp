#version 330 core

uniform float u_time;
uniform vec2 u_resolution;
uniform sampler2D u_prev_frame;
uniform vec4 u_audio;
uniform float u_feedback;
out vec4 FragColor;

// Plasma generation functions
vec3 plasma(vec2 p, float time) {
    float v1 = sin(p.x * 10.0 + time);
    float v2 = sin(10.0 * (p.x * sin(time / 2.0) + p.y * cos(time / 3.0)) + time);
    float v3 = sin(sqrt(100.0 * (p.x * p.x + p.y * p.y) + 1.0) + time);
    float v = v1 + v2 + v3;
    return vec3(sin(v), sin(v + 2.0), sin(v + 4.0)) * 0.5 + 0.5;
}

// Noise function
float noise(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    vec2 uv = gl_FragCoord.xy / u_resolution;
    vec2 p = uv * 2.0 - 1.0;
    
    // Audio-reactive plasma
    float plasma_scale = 1.0 + u_audio.x * 0.5 + u_audio.w * 0.3;
    vec3 color = plasma(p * plasma_scale, u_time * 2.0);
    
    // Add noise for texture
    float n = noise(p * 50.0 + u_time) * 0.1;
    color += n;
    
    // Beat-reactive pulse
    if (u_audio.w > 0.6) {
        float pulse = (u_audio.w - 0.6) * 2.5;
        color *= (1.0 + pulse * 0.5);
    }
    
    // Color shifting based on audio
    float hue_shift = u_time * 0.1 + u_audio.y * 0.3;
    color.r = sin(color.r * 3.14159 + hue_shift) * 0.5 + 0.5;
    color.g = sin(color.g * 3.14159 + hue_shift + 2.094) * 0.5 + 0.5;
    color.b = sin(color.b * 3.14159 + hue_shift + 4.189) * 0.5 + 0.5;
    
    // Feedback for smooth transitions
    vec4 feedback = texture(u_prev_frame, uv);
    color = mix(color, feedback.rgb, u_feedback * 0.4);
    
    // Audio enhancement
    color += vec3(u_audio.x * 0.2, u_audio.y * 0.15, u_audio.z * 0.1);
    
    FragColor = vec4(color, 1.0);
}
