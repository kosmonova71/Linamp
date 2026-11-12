#version 330 core
// Side Transition - Slide from side

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
    
    // Direction selection
    int direction = abs(iRandStatic.x) % 4;
    
    vec2 from_uv = uv;
    vec2 to_uv = uv;
    
    float offset = t;
    
    if (direction == 0) { // left to right
        from_uv.x = uv.x + offset;
        to_uv.x = uv.x - (1.0 - offset);
    } else if (direction == 1) { // right to left
        from_uv.x = uv.x - offset;
        to_uv.x = uv.x + (1.0 - offset);
    } else if (direction == 2) { // top to bottom
        from_uv.y = uv.y - offset;
        to_uv.y = uv.y + (1.0 - offset);
    } else { // bottom to top
        from_uv.y = uv.y + offset;
        to_uv.y = uv.y - (1.0 - offset);
    }
    
    // Add wave distortion at edge
    float wave = sin(uv.y * 20.0 + iTime * 5.0) * 0.02 * iBass;
    from_uv.x += wave * (1.0 - t);
    to_uv.x += wave * t;
    
    // Sample both textures
    vec4 from_color = texture(u_texture_from, from_uv);
    vec4 to_color = texture(u_texture_to, to_uv);
    
    // Mix with soft edge
    float mix_factor = smoothstep(t - 0.1, t + 0.1, offset);
    mix_factor += iMid * 0.1;
    mix_factor = clamp(mix_factor, 0.0, 1.0);
    
    FragColor = mix(from_color, to_color, mix_factor);
}
