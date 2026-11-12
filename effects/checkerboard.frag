#version 330 core
// Checkerboard Transition - Checkerboard pattern wipe

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
    
    // Checkerboard grid
    float checker_size = 8.0;
    vec2 grid = floor(uv * checker_size);
    float checker = mod(grid.x + grid.y, 2.0);
    
    // Animated reveal
    float reveal = fract(length(uv - 0.5) * 2.0 - iTime * 0.5 + iBass);
    
    // Combine checker with reveal
    float pattern = mix(checker, 1.0 - checker, step(reveal, t));
    
    // Sample both textures
    vec4 from_color = texture(u_texture_from, uv);
    vec4 to_color = texture(u_texture_to, uv);
    
    float edge_width = 0.05 + iMidAtt * 0.03;
    float mix_factor = smoothstep(t - edge_width, t + edge_width, pattern);
    
    FragColor = mix(from_color, to_color, mix_factor);
}
