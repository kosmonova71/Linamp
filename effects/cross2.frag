#version 330 core
// Cross 2 Transition - Double cross/diagonal cross

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
    
    vec2 center = vec2(0.5);
    vec2 p = uv - center;
    vec2 ap = abs(p);
    
    // Standard cross
    float bar_width = 0.1 + iMid * 0.03;
    float cross1 = 1.0 - smoothstep(0.0, bar_width, ap.x);
    cross1 = max(cross1, 1.0 - smoothstep(0.0, bar_width, ap.y));
    
    // Diagonal cross (rotated 45 degrees)
    vec2 dp = vec2(
        abs(p.x * 0.707 + p.y * 0.707),
        abs(p.x * 0.707 - p.y * 0.707)
    );
    float cross2 = 1.0 - smoothstep(0.0, bar_width, dp.x);
    cross2 = max(cross2, 1.0 - smoothstep(0.0, bar_width, dp.y));
    
    // Combine both crosses
    float cross = max(cross1, cross2);
    
    // Expanding from center with rotation
    float dist = length(p);
    float angle = atan(p.y, p.x);
    float expand = smoothstep(t - 0.15, t, dist + sin(angle * 4.0 + iTime + iBass * 2.0) * 0.05);
    
    // Sample both textures
    vec4 from_color = texture(u_texture_from, uv);
    vec4 to_color = texture(u_texture_to, uv);
    
    float mix_factor = smoothstep(t - 0.08, t + 0.08, cross * expand);
    
    FragColor = mix(from_color, to_color, mix_factor);
}
