#version 330 core
// Sliding Transition
// New preset slides in from a direction
// Direction can be: left, right, up, down

in vec2 v_uv;
out vec4 FragColor;

uniform sampler2D u_texture1;
uniform sampler2D u_texture2;

// Transition uniforms
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

// Configuration: slide direction (0=left, 1=right, 2=up, 3=down)
// Can be overridden via iRandStatic.x for randomization
const int SLIDE_DIRECTION = 0;

void main() {
    vec2 uv = v_uv;
    
    // Determine direction (use static random if available, otherwise default)
    int direction = (iRandStatic.x != 0) ? abs(iRandStatic.x) % 4 : SLIDE_DIRECTION;
    
    float t = iProgressBicubic;
    
    // Add subtle bounce effect based on bass
    float bounce = iBassAtt * 0.02 * sin(t * 3.14159 * 2.0);
    t = t + bounce;
    
    vec2 from_uv = uv;
    vec2 to_uv = uv;
    
    // Calculate offset based on direction
    if (direction == 0) {  // Left to right
        from_uv.x = uv.x + t;
        to_uv.x = uv.x - (1.0 - t);
    } else if (direction == 1) {  // Right to left
        from_uv.x = uv.x - t;
        to_uv.x = uv.x + (1.0 - t);
    } else if (direction == 2) {  // Bottom to top
        from_uv.y = uv.y + t;
        to_uv.y = uv.y - (1.0 - t);
    } else {  // Top to bottom
        from_uv.y = uv.y - t;
        to_uv.y = uv.y + (1.0 - t);
    }
    
    // Sample with clamping to avoid edge artifacts
    vec4 from_color = vec4(0.0);
    vec4 to_color = vec4(0.0);
    
    if (from_uv.x >= 0.0 && from_uv.x <= 1.0 && from_uv.y >= 0.0 && from_uv.y <= 1.0) {
        from_color = texture(u_texture1, from_uv);
    }
    if (to_uv.x >= 0.0 && to_uv.x <= 1.0 && to_uv.y >= 0.0 && to_uv.y <= 1.0) {
        to_color = texture(u_texture2, to_uv);
    }
    
    // Blend in the overlap region with a smooth edge
    float edge_width = 0.05 + iMid * 0.02;
    float mix_factor = smoothstep(0.0, edge_width, t) * (1.0 - smoothstep(1.0 - edge_width, 1.0, t));
    
    // Audio-reactive blend
    mix_factor = mix_factor * (1.0 + iBass * 0.1);
    mix_factor = clamp(mix_factor, 0.0, 1.0);
    
    // If we're fully transitioned, show only to_color
    if (t >= 0.99) {
        FragColor = to_color;
    } else if (t <= 0.01) {
        FragColor = from_color;
    } else {
        // In the middle region, blend based on position
        float slide_mix = t;
        
        // Add wave distortion at the transition edge
        float wave = sin(uv.y * 20.0 + iTime * 5.0) * iBass * 0.01;
        slide_mix += wave * (1.0 - abs(t - 0.5) * 2.0);
        slide_mix = clamp(slide_mix, 0.0, 1.0);
        
        FragColor = mix(from_color, to_color, slide_mix);
    }
}
