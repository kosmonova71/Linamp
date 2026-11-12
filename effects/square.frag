#version 330 core
// Square Transition - Square/rectangular wipe

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
    
    // Expand from center as a square
    vec2 center = vec2(0.5);
    vec2 p = abs(uv - center) * 2.0;
    float square = max(p.x, p.y);
    
    // Rotate square
    float rot = iTime * 0.3 + iBass * 0.5;
    vec2 rp = p;
    rp.x = p.x * cos(rot) - p.y * sin(rot);
    rp.y = p.x * sin(rot) + p.y * cos(rot);
    float square_rot = max(abs(rp.x), abs(rp.y));
    
    square = mix(square, square_rot, 0.3);
    
    // Sample both textures
    vec4 from_color = texture(u_texture_from, uv);
    vec4 to_color = texture(u_texture_to, uv);
    
    float edge_width = 0.03 + iMid * 0.02;
    float mix_factor = smoothstep(t - edge_width, t + edge_width, square);
    
    FragColor = mix(from_color, to_color, mix_factor);
}
