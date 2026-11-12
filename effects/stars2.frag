#version 330 core
// Stars 2 Transition - Star field wipe

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

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

void main() {
    vec2 uv = v_uv;
    float t = iProgressBicubic;
    
    // Star field pattern
    float star_field = 0.0;
    float grid_size = 10.0;
    vec2 grid_uv = uv * grid_size;
    vec2 cell = floor(grid_uv);
    vec2 cell_uv = fract(grid_uv);
    
    float h = hash(cell);
    
    // Star at center of cell
    float dist = length(cell_uv - 0.5);
    float star = 1.0 - smoothstep(0.0, 0.3 + iMid * 0.1, dist);
    
    // Star intensity based on hash
    star *= step(0.7, h);
    
    // Twinkle
    star *= 0.5 + 0.5 * sin(iTime * 5.0 + h * 10.0);
    
    star_field = star;
    
    // Reveal pattern
    float reveal = fract(h + t);
    
    // Sample both textures
    vec4 from_color = texture(u_texture_from, uv);
    vec4 to_color = texture(u_texture_to, uv);
    
    // Mix based on stars and transition progress
    float mix_factor = step(reveal, t) + star_field * 0.3 * iTreb;
    mix_factor = clamp(mix_factor, 0.0, 1.0);
    
    FragColor = mix(from_color, to_color, mix_factor);
}
