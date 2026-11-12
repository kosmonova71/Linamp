#version 330 core
// Patches Transition - Irregular patchwork wipe

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

float noise(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

float smooth_noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    
    float a = noise(i);
    float b = noise(i + vec2(1.0, 0.0));
    float c = noise(i + vec2(0.0, 1.0));
    float d = noise(i + vec2(1.0, 1.0));
    
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

void main() {
    vec2 uv = v_uv;
    float t = iProgressBicubic;
    
    // Irregular patches using noise
    float patch_scale = 8.0;
    float n1 = smooth_noise(uv * patch_scale);
    float n2 = smooth_noise(uv * patch_scale * 2.0 + vec2(iTime * 0.2));
    float n3 = smooth_noise(uv * patch_scale * 0.5 + vec2(iBass));
    
    float pattern = n1 * 0.5 + n2 * 0.3 + n3 * 0.2;
    
    // Animate patches
    pattern += iTreb * 0.2 * sin(uv.x * 10.0 + iTime * 3.0);
    
    // Threshold for patch reveal
    float threshold = t;
    float edge_width = 0.1 + iMid * 0.05;
    
    // Sample both textures
    vec4 from_color = texture(u_texture_from, uv);
    vec4 to_color = texture(u_texture_to, uv);
    
    float mix_factor = smoothstep(threshold - edge_width, threshold + edge_width, pattern);
    
    FragColor = mix(from_color, to_color, mix_factor);
}
