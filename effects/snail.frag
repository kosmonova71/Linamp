#version 330 core
// Snail/Spiral Transition - Spiral wipe effect

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
    
    // Convert to polar coordinates
    vec2 center = vec2(0.5);
    vec2 p = uv - center;
    float r = length(p);
    float angle = atan(p.y, p.x);
    
    // Spiral pattern
    float spiral = (r * 3.0 + angle / 6.28318);
    spiral = fract(spiral);
    
    // Add rotation during transition
    spiral += iTime * 0.2 + iBass * 0.3;
    spiral = fract(spiral);
    
    // Sample both textures
    vec4 from_color = texture(u_texture_from, uv);
    vec4 to_color = texture(u_texture_to, uv);
    
    // Mix based on spiral pattern
    float edge_width = 0.1 + iTreb * 0.05;
    float mix_factor = smoothstep(t - edge_width, t + edge_width, spiral);
    
    FragColor = mix(from_color, to_color, mix_factor);
}
