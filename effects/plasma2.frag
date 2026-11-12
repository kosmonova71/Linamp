#version 330 core
// Plasma 2 Transition - Animated plasma effect between presets

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
    
    // Plasma animation
    float plasma = 0.0;
    vec2 p = uv * 3.0;
    
    for(int i = 1; i < 6; i++) {
        p.x += 0.7 / i * sin(i * 3.0 * p.y + iTime * 0.5 + iBass * 2.0);
        p.y += 0.7 / i * cos(i * 3.0 * p.x + iTime * 0.3 + iTreb * 2.0);
    }
    
    plasma = 0.5 + 0.5 * sin(p.x * 2.0 + iTime);
    plasma += 0.5 + 0.5 * sin(p.y * 2.0 + iTime);
    plasma *= 0.5;
    
    // Add audio reactivity
    plasma += iBass * 0.2 * sin(uv.x * 10.0 + iTime * 5.0);
    plasma = clamp(plasma, 0.0, 1.0);
    
    // Sample both textures
    vec4 from_color = texture(u_texture_from, uv);
    vec4 to_color = texture(u_texture_to, uv);
    
    // Mix based on plasma pattern and transition progress
    float mix_factor = smoothstep(t - 0.2, t + 0.2, plasma);
    mix_factor = clamp(mix_factor + iMid * 0.1, 0.0, 1.0);
    
    FragColor = mix(from_color, to_color, mix_factor);
}
