#version 330 core
// Rotating Wipe Transition
// Like a clock hand sweeping across the screen

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
    
    // Center of the screen
    vec2 center = vec2(0.5);
    vec2 centered_uv = uv - center;
    
    // Calculate polar coordinates
    float radius = length(centered_uv);
    float angle = atan(centered_uv.y, centered_uv.x);  // -pi to pi
    
    // Normalize angle to 0 to 2*PI range
    float normalized_angle = angle;
    if (normalized_angle < 0.0) {
        normalized_angle += 6.28318530718;  // 2 * PI
    }
    
    float t = iProgressBicubic;
    
    // Starting angle offset (can use static random for variety)
    float start_angle = float(iRandStatic.x % 360) * 0.0174533;  // Convert to radians
    
    // Audio-reactive rotation speed
    float audio_boost = iBass * 0.5 + iMid * 0.3;
    float sweep_angle = t * (6.28318530718 + audio_boost);  // Full circle + extra
    
    // Current wipe angle
    float current_angle = start_angle + sweep_angle;
    current_angle = mod(current_angle, 6.28318530718);
    
    // Calculate angle difference considering circular wrap
    float angle_diff = normalized_angle - current_angle;
    if (angle_diff < 0.0) angle_diff += 6.28318530718;
    
    // Sample both textures
    vec4 from_color = texture(u_texture_from, uv);
    vec4 to_color = texture(u_texture_to, uv);
    
    // Create soft edge for the wipe
    float edge_width = 0.15 + iTreb * 0.05;
    
    // Determine if this pixel is before or after the wipe line
    float mix_factor = 1.0 - smoothstep(0.0, edge_width, angle_diff);
    
    // Add "clock hand" glow effect at the transition edge
    float glow = exp(-angle_diff * angle_diff / (edge_width * 0.3));
    glow *= iBassAtt * 0.3;
    
    // Mix colors
    vec4 color = mix(from_color, to_color, mix_factor);
    
    // Add glow
    color.rgb += to_color.rgb * glow;
    
    FragColor = color;
}
