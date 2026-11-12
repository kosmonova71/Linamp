#version 330 core
// Cisor/Scissor Transition - Scissor cut effect

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
    
    // Scissor line moving diagonally
    float line = uv.x + uv.y;
    float cut = step(line, t * 2.0 + iBass * 0.2);
    
    // Add jagged edge
    float jagged = 0.02 * sin(uv.y * 50.0 + iTime * 10.0) * iTreb;
    cut = step(line + jagged, t * 2.0);
    
    // Alternate sides
    float side = mod(floor(uv.x * 10.0) + floor(uv.y * 10.0), 2.0);
    
    // Sample both textures
    vec4 from_color = texture(u_texture_from, uv);
    vec4 to_color = texture(u_texture_to, uv);
    
    float mix_factor = mix(1.0 - cut, cut, side);
    
    FragColor = mix(from_color, to_color, mix_factor);
}
