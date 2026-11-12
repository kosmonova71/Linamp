#version 330 core
// Corner Transition - Wipe from corner to corner

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
    
    // Select corner based on random seed
    int corner = abs(iRandStatic.x) % 4;
    vec2 from_corner;
    
    if (corner == 0) from_corner = vec2(0.0, 0.0);      // bottom-left
    else if (corner == 1) from_corner = vec2(1.0, 0.0); // bottom-right
    else if (corner == 2) from_corner = vec2(0.0, 1.0); // top-left
    else from_corner = vec2(1.0, 1.0);                  // top-right
    
    // Distance from selected corner
    float dist = length(uv - from_corner);
    float max_dist = 1.414; // diagonal
    
    // Add wave distortion
    dist += iBass * 0.1 * sin(uv.x * 10.0 + iTime * 3.0);
    dist += iTreb * 0.1 * cos(uv.y * 10.0 + iTime * 2.0);
    
    float normalized_dist = dist / max_dist;
    
    // Sample both textures
    vec4 from_color = texture(u_texture_from, uv);
    vec4 to_color = texture(u_texture_to, uv);
    
    float edge_width = 0.05 + iMid * 0.03;
    float mix_factor = smoothstep(t - edge_width, t + edge_width, normalized_dist);
    
    FragColor = mix(from_color, to_color, mix_factor);
}
