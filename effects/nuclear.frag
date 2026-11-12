#version 330 core
// Nuclear Transition - Radial burst effect

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
    float r = length(p);
    float angle = atan(p.y, p.x);
    
    // Nuclear burst rays
    float rays = 8.0;
    float ray = cos(angle * rays + iTime * 2.0);
    ray = smoothstep(0.5, 1.0, ray);
    
    // Expanding ring
    float ring = smoothstep(t - 0.1, t, r) * smoothstep(t + 0.2, t + 0.1, r);
    
    // Combine
    float pattern = max(ray * ring, smoothstep(t - 0.05, t, r));
    
    // Add audio reactivity
    pattern += iBass * 0.2 * sin(r * 20.0 - iTime * 5.0);
    pattern = clamp(pattern, 0.0, 1.0);
    
    // Sample both textures
    vec4 from_color = texture(u_texture_from, uv);
    vec4 to_color = texture(u_texture_to, uv);
    
    float mix_factor = smoothstep(t - 0.1, t + 0.1, pattern);
    
    FragColor = mix(from_color, to_color, mix_factor);
}
