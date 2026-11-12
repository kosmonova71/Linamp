#version 330 core
// Wave Transition - Wave distortion wipe

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
    
    // Wave distortion
    float waves = sin(uv.x * 20.0 + iTime * 3.0 + iBass * 5.0) * 0.1;
    waves += sin(uv.y * 15.0 + iTime * 2.0) * 0.05 * iTreb;
    
    // Moving wave front
    float wave_front = uv.y + waves;
    wave_front += iBassAtt * 0.2;
    
    // Sample both textures with distortion
    vec2 from_uv = uv + vec2(0.0, waves * (1.0 - t));
    vec2 to_uv = uv + vec2(0.0, waves * t);
    
    vec4 from_color = texture(u_texture_from, from_uv);
    vec4 to_color = texture(u_texture_to, to_uv);
    
    float edge_width = 0.05 + iMid * 0.02;
    float mix_factor = smoothstep(t - edge_width, t + edge_width, wave_front);
    
    FragColor = mix(from_color, to_color, mix_factor);
}
