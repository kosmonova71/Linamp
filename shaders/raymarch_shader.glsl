#version 330 core

uniform float u_time;
uniform vec2 u_resolution;
uniform sampler2D u_prev_frame;
uniform vec4 u_audio;
uniform float u_feedback;
out vec4 FragColor;

// Raymarching functions
float sdSphere(vec3 p, float r) {
    return length(p) - r;
}

float sdBox(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float map(vec3 p) {
    float sphere = sdSphere(p, 0.5);
    float box = sdBox(p - vec3(sin(u_time) * 0.3, cos(u_time * 0.7) * 0.2, 0.0), vec3(0.3));
    return min(sphere, box);
}

vec3 getNormal(vec3 p) {
    vec2 e = vec2(0.001, 0.0);
    return normalize(vec3(
        map(p + e.xyy) - map(p - e.xyy),
        map(p + e.yxy) - map(p - e.yxy),
        map(p + e.yyx) - map(p - e.yyx)
    ));
}

float raymarch(vec3 ro, vec3 rd) {
    float t = 0.0;
    for(int i = 0; i < 100; i++) {
        vec3 p = ro + rd * t;
        float d = map(p);
        if(d < 0.001) return t;
        if(t > 10.0) break;
        t += d;
    }
    return t;
}

void main() {
    vec2 uv = gl_FragCoord.xy / u_resolution;
    uv = uv * 2.0 - 1.0;
    
    // Camera setup
    vec3 ro = vec3(0.0, 0.0, 2.0);
    vec3 rd = normalize(vec3(uv, -1.5));
    
    // Audio-reactive camera movement
    ro.x += sin(u_time * 0.5) * u_audio.x * 0.5;
    ro.y += cos(u_time * 0.3) * u_audio.y * 0.3;
    
    // Raymarch
    float t = raymarch(ro, rd);
    
    vec3 color = vec3(0.0);
    if(t < 10.0) {
        vec3 p = ro + rd * t;
        vec3 normal = getNormal(p);
        
        // Lighting
        vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
        float diff = max(dot(normal, lightDir), 0.0);
        
        // Audio-reactive colors
        color = vec3(u_audio.x, u_audio.y, u_audio.z) * diff;
        
        // Beat-reactive glow
        if(u_audio.w > 0.5) {
            float glow = (u_audio.w - 0.5) * 2.0;
            color += vec3(glow * 0.5, glow * 0.3, glow * 0.8);
        }
    }
    
    // Background gradient
    vec3 bg = mix(vec3(0.1, 0.05, 0.2), vec3(0.05, 0.1, 0.3), uv.y);
    color = mix(bg, color, step(t, 10.0));
    
    // Feedback for continuity
    vec4 feedback = texture(u_prev_frame, uv);
    color = mix(color, feedback.rgb, u_feedback * 0.2);
    
    FragColor = vec4(color, 1.0);
}
