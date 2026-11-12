#version 330 core
// Cercle/Circle Transition - Circle wipe with distortions

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
    
    // Expanding circle with ripple
    float ripple = 0.02 * sin(r * 40.0 - iTime * 5.0 + iBass * 5.0);
    float dist = r + ripple;
    
    // Multiple circles
    float circles = sin(r * 20.0 + t * 6.28318 + iTreb * 3.0);
    circles = smoothstep(0.0, 0.5, circles);
    
    // Combine
    float mask = smoothstep(t - 0.15, t + 0.15, dist + ripple);
    mask = mix(mask, circles, 0.3);
    
    // Sample both textures
    vec4 from_color = texture(u_texture_from, uv);
    vec4 to_color = texture(u_texture_to, uv);
    
    float mix_factor = clamp(mask + iMid * 0.1, 0.0, 1.0);
    
    FragColor = mix(from_color, to_color, mix_factor);
}
