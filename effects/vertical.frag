#version 330 core
// Vertical Wipe Transition - Vertical line wipe

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
    
    // Vertical wipe with jagged edge
    float jagged = 0.02 * sin(uv.y * 40.0 + iTime * 8.0) * iBass;
    float wipe = uv.x + jagged;
    
    // Direction selection
    int direction = abs(iRandStatic.x) % 2;
    if (direction == 1) {
        wipe = 1.0 - uv.x + jagged;
    }
    
    // Sample both textures
    vec4 from_color = texture(u_texture_from, uv);
    vec4 to_color = texture(u_texture_to, uv);
    
    float edge_width = 0.02 + iMid * 0.02;
    float mix_factor = smoothstep(t - edge_width, t + edge_width, wipe);
    
    FragColor = mix(from_color, to_color, mix_factor);
}
