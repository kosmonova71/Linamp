#version 330 core
// Wave Distortion Transition
// Distorts the transition boundary with wave patterns

in vec2 v_uv;
out vec4 FragColor;

uniform sampler2D u_texture_from;
uniform sampler2D u_texture_to;

// Transition uniforms
uniform float iProgressLinear;
uniform float iProgressCosine;
uniform float iProgressBicubic;
uniform float iTransitionDuration;
uniform float iTime;
uniform float iTimeDelta;
uniform float iFrameRate;
uniform int iFrame;
uniform ivec4 iRandStatic;
uniform ivec4 iRandFrame;
uniform float iBass;
uniform float iMid;
uniform float iTreb;
uniform float iBassAtt;
uniform float iMidAtt;
uniform float iTrebAtt;

uniform vec2 fragCoord;
uniform vec3 iResolution;

void main() {
    vec2 uv = v_uv;
    float t = iProgressBicubic;
    
    // Direction of wave (0=horizontal, 1=vertical)
    int wave_dir = (iRandStatic.y != 0) ? abs(iRandStatic.y) % 2 : 0;
    
    // Base transition coordinate
    float coord = (wave_dir == 0) ? uv.x : uv.y;
    
    // Create multi-frequency wave distortion
    float wave1 = sin(coord * 10.0 + iTime * 3.0) * (0.02 + iBass * 0.01);
    float wave2 = sin(coord * 25.0 + iTime * 5.0) * (0.01 + iMid * 0.005);
    float wave3 = sin(coord * 40.0 - iTime * 4.0) * (0.005 + iTreb * 0.003);
    
    float total_wave = wave1 + wave2 + wave3;
    
    // Secondary wave perpendicular to main wave
    float perp_coord = (wave_dir == 0) ? uv.y : uv.x;
    float perp_wave = sin(perp_coord * 15.0 + iTime * 2.0) * 0.02 * iBassAtt;
    
    // Apply wave distortion to the transition boundary
    float distorted_t = t + total_wave + perp_wave;
    distorted_t = clamp(distorted_t, 0.0, 1.0);
    
    // Sample with distorted UVs for ripple effect
    vec2 distorted_uv = uv;
    float ripple = sin(t * 3.14159 * 2.0) * total_wave * 2.0;
    
    if (wave_dir == 0) {
        distorted_uv.x += ripple * (1.0 - abs(t - 0.5) * 2.0);
    } else {
        distorted_uv.y += ripple * (1.0 - abs(t - 0.5) * 2.0);
    }
    
    // Clamp UVs
    distorted_uv = clamp(distorted_uv, 0.0, 1.0);
    
    // Sample both textures
    vec4 from_color = texture(u_texture_from, distorted_uv);
    vec4 to_color = texture(u_texture_to, distorted_uv);
    
    // Create smooth transition with wave-affected edge
    float edge_softness = 0.1 + iMidAtt * 0.05;
    
    // The actual transition boundary follows the waves
    float boundary = distorted_t;
    float mix_factor = smoothstep(boundary - edge_softness, boundary + edge_softness, coord);
    
    // Add color separation (chromatic aberration) at wave peaks
    float ca_strength = abs(total_wave) * 5.0 * iTreb;
    
    vec4 color = mix(from_color, to_color, mix_factor);
    
    // Apply chromatic aberration in transition zone
    if (ca_strength > 0.001 && abs(coord - boundary) < edge_softness * 2.0) {
        vec2 ca_offset = vec2(ca_strength * 0.01, 0.0);
        if (wave_dir == 1) ca_offset = vec2(0.0, ca_strength * 0.01);
        
        float r = texture(u_texture_from, clamp(distorted_uv + ca_offset, 0.0, 1.0)).r;
        float b = texture(u_texture_from, clamp(distorted_uv - ca_offset, 0.0, 1.0)).b;
        
        vec4 ca_color = vec4(r, from_color.g, b, from_color.a);
        color = mix(color, ca_color, abs(coord - boundary) / (edge_softness * 2.0));
    }
    
    // Add glow at wave crests
    float glow = smoothstep(0.02, 0.0, abs(total_wave)) * iBass * 0.3;
    color.rgb += glow;
    
    FragColor = color;
}
