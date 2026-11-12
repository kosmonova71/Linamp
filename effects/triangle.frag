#version 330 core
// Triangle Transition - Triangular wipe pattern

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

float triangle_pattern(vec2 uv) {
    // Create triangle grid
    vec2 grid = uv * 6.0;
    vec2 cell = fract(grid);
    
    float d1 = cell.x;
    float d2 = cell.y;
    float d3 = 1.0 - cell.x - cell.y;
    
    return max(max(d1, d2), -d3);
}

void main() {
    vec2 uv = v_uv;
    float t = iProgressBicubic;
    
    // Triangle pattern
    float tri = triangle_pattern(uv + vec2(iTime * 0.1, iTime * 0.05));
    
    // Add audio reactivity
    tri += iBass * 0.2 * sin(uv.x * 15.0 + iTime * 4.0);
    tri = fract(tri);
    
    // Sample both textures
    vec4 from_color = texture(u_texture_from, uv);
    vec4 to_color = texture(u_texture_to, uv);
    
    float mix_factor = smoothstep(t - 0.1, t + 0.1, tri);
    
    FragColor = mix(from_color, to_color, mix_factor);
}
