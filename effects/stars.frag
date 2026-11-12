#version 330 core
// Stars Transition - Star burst wipe effect

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
    
    // Star pattern (5 pointed)
    vec2 center = vec2(0.5);
    vec2 p = uv - center;
    float r = length(p);
    float angle = atan(p.y, p.x);
    
    // Create star shape
    float star = abs(cos(angle * 5.0));
    star = pow(star, 2.0);
    star = 1.0 - star * (0.5 - r);
    
    // Rotate star
    angle += iTime * 0.5 + iBass;
    float star2 = abs(cos(angle * 5.0));
    star2 = pow(star2, 2.0);
    
    // Combine
    float pattern = mix(star, star2, 0.5 + 0.5 * sin(iTime + iTreb * 3.0));
    
    // Sample both textures
    vec4 from_color = texture(u_texture_from, uv);
    vec4 to_color = texture(u_texture_to, uv);
    
    float edge_width = 0.1 + iMid * 0.05;
    float mix_factor = smoothstep(t - edge_width, t + edge_width, pattern);
    
    FragColor = mix(from_color, to_color, mix_factor);
}
