#version 330 core
// Cross Transition - Cross/plus wipe

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
    vec2 p = abs(uv - center);
    
    // Cross shape (horizontal + vertical bars)
    float bar_width = 0.15 + iMid * 0.05;
    float h_bar = 1.0 - smoothstep(0.0, bar_width, p.y);
    float v_bar = 1.0 - smoothstep(0.0, bar_width, p.x);
    float cross = max(h_bar, v_bar);
    
    // Expanding from center
    float dist = length(p);
    float expand = smoothstep(t - 0.1, t, dist) * cross;
    
    // Add ripple
    float ripple = 0.02 * sin(dist * 40.0 - iTime * 5.0 + iBass * 5.0);
    
    // Sample both textures
    vec4 from_color = texture(u_texture_from, uv);
    vec4 to_color = texture(u_texture_to, uv);
    
    float mix_factor = smoothstep(t - 0.05, t + 0.05, expand + ripple);
    
    FragColor = mix(from_color, to_color, mix_factor);
}
