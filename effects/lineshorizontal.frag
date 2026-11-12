#version 330 core
// Lines Horizontal Transition - Multiple horizontal lines wipe

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
    
    // Multiple horizontal lines
    float num_lines = 12.0;
    float line_height = 1.0 / num_lines;
    float line_index = floor(uv.y * num_lines);
    
    // Staggered wipe
    float stagger = fract(line_index * 0.618);
    float wipe = uv.y + cos(stagger * 6.28318) * 0.1;
    
    // Add wave to each line
    wipe += iTreb * 0.02 * sin(uv.x * 20.0 + iTime * 4.0 + line_index);
    
    // Sample both textures
    vec4 from_color = texture(u_texture_from, uv);
    vec4 to_color = texture(u_texture_to, uv);
    
    float edge_width = line_height * 0.3 + iMid * 0.01;
    float mix_factor = smoothstep(t - edge_width, t + edge_width, wipe);
    
    FragColor = mix(from_color, to_color, mix_factor);
}
