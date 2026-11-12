#version 330 core
// Bubbles Transition - Rising bubble reveal

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

float circle(vec2 uv, vec2 pos, float r) {
    return smoothstep(r, r - 0.02, length(uv - pos));
}

void main() {
    vec2 uv = v_uv;
    float t = iProgressBicubic;
    
    // Generate bubbles
    float bubbles = 0.0;
    for(int i = 0; i < 20; i++) {
        float seed = i * 1.618;
        vec2 pos = vec2(
            fract(sin(seed) * 43758.5453),
            fract(sin(seed + 1.0) * 43758.5453)
        );
        
        // Animate bubbles rising
        pos.y = fract(pos.y + iTime * 0.1 * (1.0 + i * 0.1) + iBass * 0.3);
        
        float size = 0.03 + 0.04 * fract(sin(seed + 2.0) * 43758.5453) * (1.0 + iMid);
        bubbles += circle(uv, pos, size);
    }
    
    bubbles = clamp(bubbles, 0.0, 1.0);
    bubbles = 1.0 - bubbles; // Invert so bubbles show new preset
    
    // Sample both textures
    vec4 from_color = texture(u_texture_from, uv);
    vec4 to_color = texture(u_texture_to, uv);
    
    float mix_factor = smoothstep(t - 0.1, t + 0.1, bubbles);
    
    FragColor = mix(from_color, to_color, mix_factor);
}
