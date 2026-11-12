#version 330 core
// Clock Transition - Clock hand wipe

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
    float angle = atan(p.y, p.x);
    
    // Normalize angle to 0-1
    float normalized_angle = (angle + 3.14159) / (2.0 * 3.14159);
    
    // Clock sweep
    float sweep = fract(normalized_angle - iTime * 0.1);
    
    // Add clock hand width
    float hand_width = 0.02 + iMid * 0.01;
    float dist = length(p);
    
    // Create clock wipe
    float clock = step(sweep, t);
    
    // Add radial ripple
    float ripple = 0.05 * sin(dist * 30.0 - iTime * 5.0 + iBass * 5.0);
    
    // Sample both textures
    vec4 from_color = texture(u_texture_from, uv + vec2(ripple * (1.0 - t)));
    vec4 to_color = texture(u_texture_to, uv + vec2(ripple * t));
    
    float mix_factor = smoothstep(t - hand_width, t + hand_width, sweep);
    
    FragColor = mix(from_color, to_color, mix_factor);
}
