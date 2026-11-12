#version 330 core

// Bloom and Glow Post-Processing Shader
uniform sampler2D u_scene;      // Main scene texture
uniform sampler2D u_bloom;       // Bloom texture
uniform float u_bloom_strength; // Bloom intensity (0.0 - 1.0)
uniform float u_glow_amount;    // Glow amount
uniform float u_time;           // Time for animated effects
uniform vec2 u_resolution;      // Screen resolution

in vec2 v_texcoord;
out vec4 FragColor;

// Simple blur function for bloom
vec3 blur(sampler2D tex, vec2 uv, vec2 texelSize) {
    vec3 color = vec3(0.0);
    float total = 0.0;
    
    // 9-tap Gaussian blur
    float weights[9] = float[](
        1.0, 2.0, 1.0,
        2.0, 4.0, 2.0,
        1.0, 2.0, 1.0
    );
    
    vec2 offsets[9] = vec2[](
        vec2(-1, -1), vec2(0, -1), vec2(1, -1),
        vec2(-1,  0), vec2(0,  0), vec2(1,  0),
        vec2(-1,  1), vec2(0,  1), vec2(1,  1)
    );
    
    for(int i = 0; i < 9; i++) {
        vec2 sample_uv = uv + offsets[i] * texelSize;
        color += texture(tex, sample_uv).rgb * weights[i];
        total += weights[i];
    }
    
    return color / total;
}

// Vignette effect
float vignette(vec2 uv) {
    vec2 center = vec2(0.5, 0.5);
    float dist = length(uv - center);
    return 1.0 - smoothstep(0.3, 0.8, dist);
}

// Color grading
vec3 colorGrade(vec3 color) {
    // Contrast enhancement
    color = (color - 0.5) * 1.2 + 0.5;
    
    // Saturation boost
    float gray = dot(color, vec3(0.299, 0.587, 0.114));
    color = mix(vec3(gray), color, 1.2);
    
    // Subtle color temperature adjustment
    color.r *= 1.05;
    color.b *= 0.95;
    
    return clamp(color, 0.0, 1.0);
}

// Animated glow effect
vec3 animatedGlow(vec3 color, vec2 uv) {
    // Create subtle pulsing glow
    float pulse = sin(u_time * 2.0) * 0.1 + 0.9;
    
    // Add chromatic aberration for glow
    vec2 ca_offset = vec2(0.002, 0.001) * u_glow_amount;
    float r = texture(u_scene, uv + ca_offset).r;
    float g = texture(u_scene, uv).g;
    float b = texture(u_scene, uv - ca_offset).b;
    
    vec3 ca_color = vec3(r, g, b);
    color = mix(color, ca_color, u_glow_amount * 0.3);
    
    return color * pulse;
}

void main() {
    vec2 uv = v_texcoord;
    vec2 texelSize = 1.0 / u_resolution;
    
    // Sample main scene
    vec3 scene_color = texture(u_scene, uv).rgb;
    
    // Sample and blur bloom texture
    vec3 bloom_color = blur(u_bloom, uv, texelSize * 2.0);
    
    // Apply bloom with intensity control
    bloom_color *= u_bloom_strength;
    
    // Combine scene with bloom
    vec3 color = scene_color + bloom_color;
    
    // Apply animated glow
    color = animatedGlow(color, uv);
    
    // Apply vignette
    float vignette_factor = vignette(uv);
    color *= vignette_factor;
    
    // Color grading
    color = colorGrade(color);
    
    // Final gamma correction
    color = pow(color, vec3(0.95));
    
    FragColor = vec4(color, 1.0);
}
