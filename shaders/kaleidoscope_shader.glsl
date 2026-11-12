#version 330 core

uniform float u_time;
uniform vec2 u_resolution;
uniform sampler2D u_prev_frame;
uniform vec4 u_audio;
uniform float u_feedback;
out vec4 FragColor;

// Kaleidoscope transformation
vec2 kaleidoscope(vec2 p, float segments) {
    float angle = atan(p.y, p.x);
    float radius = length(p);
    
    // Create kaleidoscope segments
    angle = mod(angle, 3.14159 / segments);
    if(mod(angle * segments / 3.14159, 2.0) > 1.0) {
        angle = -angle;
    }
    
    return vec2(cos(angle), sin(angle)) * radius;
}

// Noise function
float noise(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

// HSV to RGB
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    vec2 uv = gl_FragCoord.xy / u_resolution;
    vec2 p = uv * 2.0 - 1.0;
    
    // Audio-reactive kaleidoscope segments
    float segments = 6.0 + u_audio.x * 4.0 + u_audio.w * 2.0;
    p = kaleidoscope(p, segments);
    
    // Create pattern
    float pattern = sin(p.x * 20.0 + u_time * 2.0) * cos(p.y * 20.0 + u_time * 1.5);
    pattern += sin(length(p) * 30.0 - u_time * 3.0);
    pattern = pattern * 0.5 + 0.5;
    
    // Add noise texture
    float n = noise(p * 50.0 + u_time);
    pattern += n * 0.1;
    
    // Dynamic coloring
    float hue = fract(u_time * 0.15 + length(p) + u_audio.y * 0.3);
    float sat = 0.7 + 0.3 * sin(u_time * 2.0 + u_audio.z * 3.0);
    float val = pattern * 0.8 + 0.2;
    
    vec3 color = hsv2rgb(vec3(hue, sat, val));
    
    // Audio enhancement
    color += vec3(u_audio.x * 0.3, u_audio.y * 0.2, u_audio.z * 0.25);
    
    // Beat-reactive flash
    if(u_audio.w > 0.7) {
        float flash = (u_audio.w - 0.7) * 3.33;
        color += vec3(flash * 0.5, flash * 0.4, flash * 0.6);
    }
    
    // Feedback for smooth transitions
    vec4 feedback = texture(u_prev_frame, uv);
    color = mix(color, feedback.rgb, u_feedback * 0.5);
    
    FragColor = vec4(color, 1.0);
}
