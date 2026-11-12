#version 330 core

uniform float u_time;
uniform vec2 u_resolution;
uniform sampler2D u_prev_frame;
uniform vec4 u_audio;
uniform float u_feedback;
out vec4 FragColor;

// Particle system simulation
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

float random(vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}

void main() {
    vec2 uv = gl_FragCoord.xy / u_resolution;
    vec2 p = uv * 2.0 - 1.0;
    
    // Create particle field
    vec3 color = vec3(0.0);
    float particle_count = 50.0 + u_audio.x * 100.0;
    
    for(int i = 0; i < 100; i++) {
        if(i >= particle_count) break;
        
        // Random particle position
        vec2 seed = vec2(i * 0.123, i * 0.456);
        vec2 pos = vec2(random(seed), random(seed + 0.1)) * 2.0 - 1.0;
        
        // Audio-reactive movement
        pos.x += sin(u_time * 2.0 + i) * u_audio.y * 0.5;
        pos.y += cos(u_time * 1.5 + i * 0.5) * u_audio.z * 0.3;
        
        // Beat-reactive explosion
        if(u_audio.w > 0.5) {
            float force = (u_audio.w - 0.5) * 3.0;
            pos += normalize(pos - vec2(0.0)) * force * 0.2;
        }
        
        // Distance to particle
        float dist = length(p - pos);
        float particle = 1.0 / (dist * dist * 20.0 + 1.0);
        
        // Particle color based on audio
        float hue = fract(i * 0.1 + u_time * 0.1 + u_audio.x * 0.2);
        vec3 pcolor = hsv2rgb(vec3(hue, 0.8, particle));
        
        color += pcolor;
    }
    
    // Add glow effect
    float glow = length(color) * 0.5;
    color += vec3(glow * 0.2, glow * 0.3, glow * 0.5);
    
    // Background
    vec3 bg = vec3(0.05, 0.02, 0.1) * (1.0 + u_audio.x * 0.2);
    color += bg;
    
    // Feedback for trails
    vec4 feedback = texture(u_prev_frame, uv);
    color = mix(color, feedback.rgb, u_feedback * 0.5);
    
    // Audio enhancement
    color += vec3(u_audio.x * 0.1, u_audio.y * 0.05, u_audio.z * 0.08);
    
    FragColor = vec4(color, 1.0);
}
