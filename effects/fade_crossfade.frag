#version 330 core
// Smooth Fade/Crossfade Transition
// Fades between two presets using a simple alpha blend

in vec2 v_uv;
out vec4 FragColor;

uniform sampler2D u_texture1;  // Previous preset
uniform sampler2D u_texture2;    // Target preset

// Transition uniforms
uniform float iProgressLinear;     // 0.0 to 1.0
uniform float iProgressCosine;     // Cosine-smoothed progress
uniform float iProgressBicubic;    // Bicubic-smoothed progress
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

// Additional uniforms
uniform vec2 fragCoord;
uniform vec3 iResolution;

void main() {
    vec2 uv = v_uv;
    
// Sample both textures
    vec4 from_color = texture(u_texture1, uv);
    vec4 to_color = texture(u_texture2, uv);
    
    // Use bicubic progress for smoother fade
    float t = iProgressBicubic;
    
    // Optional: Add subtle audio reactivity
    // Reduced audio boost to prevent volume spikes during transitions
    float audio_boost = iBass * 0.02 * sin(iTime * 5.0);
    t = clamp(t + audio_boost, 0.0, 1.0);
    
    // Simple crossfade mix
    FragColor = mix(from_color, to_color, t);
}
