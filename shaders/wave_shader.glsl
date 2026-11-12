#version 330 core

uniform float u_time;
uniform vec2 u_resolution;
uniform sampler2D u_prev_frame;
uniform vec4 u_audio;
uniform float u_feedback;
out vec4 FragColor;

// Wave interference patterns
float wave(vec2 p, float freq, float speed) {
    return sin(p.x * freq + u_time * speed) * cos(p.y * freq + u_time * speed * 0.7);
}

void main() {
    vec2 uv = gl_FragCoord.xy / u_resolution;
    vec2 p = uv * 2.0 - 1.0;
    
    // Multiple wave layers
    float w1 = wave(p, 5.0, 2.0);
    float w2 = wave(p * 1.5, 8.0, 3.0);
    float w3 = wave(p * 0.8, 12.0, 4.0);
    
    // Audio-reactive wave amplitude
    float audio_mod = 1.0 + u_audio.x * 2.0 + u_audio.w * 1.5;
    float waves = (w1 + w2 + w3) * audio_mod;
    
    // Create interference pattern
    float interference = abs(waves) * 0.3 + 0.7;
    
    // Color based on wave intensity
    vec3 color = vec3(0.0);
    color.r = interference * (0.5 + 0.5 * sin(u_time + p.x * 10.0));
    color.g = interference * (0.5 + 0.5 * sin(u_time + p.y * 10.0 + 2.094));
    color.b = interference * (0.5 + 0.5 * sin(u_time + length(p) * 15.0 + 4.189));
    
    // Audio-reactive color enhancement
    color.r += u_audio.x * 0.4;
    color.g += u_audio.y * 0.3;
    color.b += u_audio.z * 0.5;
    
    // Beat-reactive pulse
    if(u_audio.w > 0.6) {
        float pulse = (u_audio.w - 0.6) * 2.5;
        color *= (1.0 + pulse * 0.8);
    }
    
    // Add some texture
    float noise = fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
    color += noise * 0.05;
    
    // Feedback for smooth transitions
    vec4 feedback = texture(u_prev_frame, uv);
    color = mix(color, feedback.rgb, u_feedback * 0.4);
    
    FragColor = vec4(color, 1.0);
}
