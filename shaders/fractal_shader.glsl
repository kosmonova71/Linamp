#version 330 core

uniform float u_time;
uniform vec2 u_resolution;
uniform sampler2D u_prev_frame;
uniform vec4 u_audio;
uniform float u_feedback;
out vec4 FragColor;

// Fractal noise functions
vec2 complexMul(vec2 a, vec2 b) {
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

float mandelbrot(vec2 c) {
    vec2 z = vec2(0.0);
    float iter = 0.0;
    
    for(int i = 0; i < 50; i++) {
        if(length(z) > 2.0) break;
        z = complexMul(z, z) + c;
        iter += 1.0;
    }
    
    return iter / 50.0;
}

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    vec2 uv = gl_FragCoord.xy / u_resolution;
    vec2 p = uv * 2.0 - 1.0;
    
    // Audio-reactive zoom
    float zoom = 1.0 + u_audio.x * 2.0 + u_time * 0.1;
    p *= zoom;
    
    // Fractal calculation
    float m = mandelbrot(p * 0.5);
    
    // Dynamic coloring based on fractal depth
    float hue = fract(m * 3.0 + u_time * 0.2 + u_audio.y);
    float sat = 0.8 + u_audio.z * 0.2;
    float val = m * 0.8 + 0.2;
    
    vec3 color = hsv2rgb(vec3(hue, sat, val));
    
    // Audio reactive enhancement
    color += u_audio.rgb * 0.3;
    
    // Feedback for continuity
    vec4 feedback = texture(u_prev_frame, uv);
    color = mix(color, feedback.rgb, u_feedback * 0.3);
    
    FragColor = vec4(color, 1.0);
}
