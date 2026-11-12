#version 330 core
// Donuts Transition - Concentric ring wipe

in vec2 v_uv;
out vec4 FragColor;

uniform sampler2D u_texture_from;
uniform sampler2D u_texture_to;

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
    
    vec2 center = vec2(0.5);
    float r = length(uv - center);
    
    // Concentric rings
    float rings = 8.0;
    float donut = fract(r * rings - iTime * 0.3 + iBass * 0.5);
    
    // Add thickness variation
    float thickness = 0.5 + 0.3 * sin(r * 20.0 + iTime * 2.0) * iTreb;
    donut = smoothstep(0.0, thickness, donut);
    
    // Sample both textures
    vec4 from_color = texture(u_texture_from, uv);
    vec4 to_color = texture(u_texture_to, uv);
    
    float mix_factor = smoothstep(t - 0.15, t + 0.15, donut);
    
    FragColor = mix(from_color, to_color, mix_factor);
}
