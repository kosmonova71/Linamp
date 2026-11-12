#version 330 core
// Zoom Transition - Zoom in/out effect

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
    
    // Zoom from center
    vec2 center = vec2(0.5);
    
    // Zoom in effect for 'from' texture
    float zoom_from = 1.0 + t * 2.0 + iBass * 0.5;
    vec2 from_uv = (uv - center) / zoom_from + center;
    
    // Zoom out effect for 'to' texture
    float zoom_to = 3.0 - t * 2.0 + iTreb * 0.5;
    vec2 to_uv = (uv - center) / zoom_to + center;
    
    // Sample both textures
    vec4 from_color = texture(u_texture_from, from_uv);
    vec4 to_color = texture(u_texture_to, to_uv);
    
    // Blend with circular mask
    float dist = length(uv - center);
    float mask = 1.0 - smoothstep(t - 0.2, t + 0.2, dist);
    mask += iMid * 0.2 * sin(dist * 20.0 - iTime * 5.0);
    mask = clamp(mask, 0.0, 1.0);
    
    FragColor = mix(from_color, to_color, mask);
}
