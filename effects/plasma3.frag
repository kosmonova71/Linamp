#version 330 core
// Plasma 3 Transition - Colorful electric plasma transition

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
    
    // Electric plasma with color channels
    vec2 p = uv * 4.0 - vec2(2.0);
    
    float r = 0.0, g = 0.0, b = 0.0;
    
    for(int i = 1; i < 4; i++) {
        p.x += 0.5 / i * sin(i * 4.0 * p.y + iTime * 0.8 + iBass * 3.0);
        p.y += 0.5 / i * cos(i * 4.0 * p.x + iTime * 0.6 + iTreb * 3.0);
    }
    
    r = 0.5 + 0.5 * sin(p.x * 3.0 + iTime);
    g = 0.5 + 0.5 * sin(p.y * 3.0 + iTime + 2.09);
    b = 0.5 + 0.5 * sin((p.x + p.y) * 2.0 + iTime + 4.18);
    
    vec3 plasma_color = vec3(r, g, b);
    float plasma_intensity = length(plasma_color) / 1.732;
    
    // Sample both textures
    vec4 from_color = texture(u_texture_from, uv);
    vec4 to_color = texture(u_texture_to, uv);
    
    // Mix based on plasma intensity
    float mix_factor = smoothstep(t - 0.15, t + 0.15, plasma_intensity);
    
    FragColor = mix(from_color, to_color, mix_factor);
}
