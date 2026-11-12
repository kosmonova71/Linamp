#version 330 core
in vec2 v_uv;
out vec4 FragColor;
uniform sampler2D u_from;
uniform sampler2D u_to;
uniform float u_progress;
uniform vec2 u_resolution;
uniform float u_pixel_size;  // New: control pixel size (1.0 = default, >1 = bigger pixels)
uniform int u_pixel_shape;   // New: 0=square, 1=circle, 2=triangle

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

float rand(vec2 co) {
    return hash(co);
}

// Smoothstep-based pixel reveal for cleaner edges
float smooth_pixel_reveal(vec2 uv, vec2 pixel_size, int shape) {
    // Get the pixel coordinates
    vec2 pixel_uv = floor(uv / pixel_size) * pixel_size;
    
    // Position within the pixel
    vec2 pos_in_pixel = frac(uv / pixel_size);
    
    float reveal = 0.0;
    
    if (shape == 0) {
        // Square pixels - use random per pixel
        reveal = step(rand(pixel_uv), u_progress);
    } else if (shape == 1) {
        // Circular pixels - distance from center
        vec2 centered = pos_in_pixel - 0.5;
        float dist = length(centered);
        float threshold = 0.5 - 0.5 * cos(3.14159 * u_progress); // animated circle
        reveal = step(dist, threshold);
    } else if (shape == 2) {
        // Triangular pixels - linear gradient
        vec2 centered = pos_in_pixel - 0.5;
        float abs_x = abs(centered.x);
        float abs_y = abs(centered.y);
        // Triangle shape: top-left to bottom-right
        float triangle = 1.0 - abs_x - abs_y;
        reveal = step(triangle, u_progress - 0.5);
    }
    
    return reveal;
}

void main() {
    vec2 pixel_size = vec2(u_pixel_size, u_pixel_size);
    
    // Use improved pixel reveal function
    float reveal = smooth_pixel_reveal(v_uv, pixel_size, u_pixel_shape);
    
    // Sample both textures
    vec4 from_color = texture(u_from, v_uv);
    vec4 to_color = texture(u_to, v_uv);
    
    // Mix based on reveal - newer pixels from 'to' appear as progress increases
    FragColor = mix(from_color, to_color, reveal);
}