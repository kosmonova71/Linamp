#version 330 core
// Plasma Transition - Classic plasma morph

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
    
    // Classic plasma
    float v = 0.0;
    vec2 p = uv * 6.0;
    
    v += sin(p.x * 2.0 + iTime);
    v += sin(p.y * 3.0 + iTime * 0.5);
    v += sin((p.x + p.y) * 1.5 + iTime * 0.3);
    
    float cx = p.x + 0.5 * sin(iTime * 0.5) + iBass;
    float cy = p.y + 0.5 * cos(iTime * 0.3) + iTreb;
    v += sin(sqrt(cx * cx + cy * cy + 1.0) + iTime);
    
    v = v / 4.0;
    
    // Color based on plasma
    float r = 0.5 + 0.5 * sin(v * 3.14159);
    float g = 0.5 + 0.5 * sin(v * 3.14159 + 2.0);
    float b = 0.5 + 0.5 * sin(v * 3.14159 + 4.0);
    
    // Sample both textures
    vec4 from_color = texture(u_texture_from, uv);
    vec4 to_color = texture(u_texture_to, uv);
    
    // Plasma as mixing mask
    float plasma_mask = 0.5 + 0.5 * sin(v * 10.0 + t * 3.14159);
    plasma_mask += iMid * 0.2;
    plasma_mask = clamp(plasma_mask, 0.0, 1.0);
    
    FragColor = mix(from_color, to_color, plasma_mask);
}
