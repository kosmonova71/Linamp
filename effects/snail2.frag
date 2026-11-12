#version 330 core
// Snail 2 Transition - Double spiral wipe

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
    vec2 p = uv - center;
    float r = length(p);
    float angle = atan(p.y, p.x);
    
    // Double spiral
    float spiral1 = fract(r * 4.0 + angle / 6.28318 + iTime * 0.15);
    float spiral2 = fract(r * 4.0 - angle / 6.28318 - iTime * 0.15);
    float spiral = mix(spiral1, spiral2, 0.5 + 0.5 * sin(iTime + iBass * 3.0));
    
    // Sample both textures
    vec4 from_color = texture(u_texture_from, uv);
    vec4 to_color = texture(u_texture_to, uv);
    
    float edge_width = 0.08 + iMid * 0.03;
    float mix_factor = smoothstep(t - edge_width, t + edge_width, spiral);
    
    FragColor = mix(from_color, to_color, mix_factor);
}
