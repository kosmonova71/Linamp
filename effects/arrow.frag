#version 330 core
// Arrow Transition - Arrow wipe direction

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
    
    // Arrow shape
    vec2 center = vec2(0.5);
    vec2 p = uv - center;
    
    // Rotate based on random seed
    int dir = abs(iRandStatic.x) % 4;
    float rotation = float(dir) * 1.5708; // 90 degree increments
    
    float c = cos(rotation);
    float s = sin(rotation);
    vec2 rp = vec2(p.x * c - p.y * s, p.x * s + p.y * c);
    
    // Arrow head shape
    float arrow = 1.0 - abs(rp.y * 3.0 - rp.x * 1.5);
    arrow = smoothstep(0.0, 0.5, arrow);
    
    // Arrow shaft
    float shaft = 1.0 - abs(rp.y * 8.0);
    shaft = smoothstep(0.0, 0.8, shaft) * step(rp.x, 0.0);
    
    float shape = max(arrow, shaft);
    
    // Moving front
    float front = rp.x + iTime * 0.2 + iBass * 0.3;
    front = fract(front);
    
    // Sample both textures
    vec4 from_color = texture(u_texture_from, uv);
    vec4 to_color = texture(u_texture_to, uv);
    
    float mix_factor = smoothstep(t - 0.15, t + 0.15, shape * front);
    
    FragColor = mix(from_color, to_color, mix_factor);
}
