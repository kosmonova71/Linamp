#version 330 core
// Curtain Transition - Curtain opening effect

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
    
    // Curtain opens from center outward
    float curtain = abs(uv.x - 0.5) * 2.0;
    
    // Add fabric ripple effect
    float ripple = 0.05 * sin(uv.y * 30.0 + iTime * 5.0 + curtain * 10.0) * iBass;
    curtain += ripple;
    
    // Fold effect
    float fold = 0.02 * sin(uv.y * 50.0 + iTime * 3.0) * (1.0 - t) * iTreb;
    
    // Sample both textures
    vec4 from_color = texture(u_texture_from, uv + vec2(fold, 0.0));
    vec4 to_color = texture(u_texture_to, uv);
    
    float edge_width = 0.08 + iMid * 0.04;
    float mix_factor = smoothstep(t - edge_width, t + edge_width, curtain);
    
    FragColor = mix(from_color, to_color, mix_factor);
}
