#version 330 core
// Reactive Visual Intelligence System (RVIS) Shader - Enhanced
// Intelligence-Driven Fragment Shader with Advanced Visual Effects

// Core brain intelligence uniforms
uniform int u_brain_state;
uniform float u_intensity;
uniform float u_momentum;
uniform float u_stability;
uniform float u_energy;

// Derived visual parameters
uniform float u_warp_strength;
uniform float u_warp_speed;
uniform float u_color_intensity;
uniform float u_color_shift;
uniform float u_motion_speed;
uniform float u_motion_damping;
uniform float u_particle_density;
uniform float u_particle_activity;
uniform float u_glow_amount;

// Time and evolution
uniform float u_brain_time;
uniform float u_state_duration;
uniform float u_time;
uniform vec2 u_resolution;

// Textures
uniform sampler2D u_prev_frame;
uniform sampler2D u_noise_tex;

in vec2 v_texcoord;
out vec4 FragColor;

// State definitions
#define STATE_CALM 0
#define STATE_FLOW 1
#define STATE_DRIVE 2
#define STATE_OVERDRIVE 3
#define STATE_CHAOS 4

// Enhanced noise functions
float noise(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

float smoothNoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    
    float a = noise(i);
    float b = noise(i + vec2(1.0, 0.0));
    float c = noise(i + vec2(0.0, 1.0));
    float d = noise(i + vec2(1.0, 1.0));
    
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm(vec2 p) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    
    for(int i = 0; i < 6; i++) {
        value += amplitude * smoothNoise(p * frequency);
        frequency *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}

// HSV to RGB conversion
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// Advanced warp function with multiple layers
vec2 advancedWarp(vec2 p, float strength, float speed) {
    float r = length(p);
    float theta = atan(p.y, p.x);
    
    // Multi-layer warp for complexity
    float warp1 = sin(r * 3.0 + u_time * speed) * strength;
    float warp2 = sin(theta * 5.0 - u_time * speed * 0.7) * strength * 0.5;
    float warp3 = cos(r * theta * 2.0 + u_time * speed * 1.3) * strength * 0.3;
    
    // Apply momentum-based distortion
    float momentum_warp = u_momentum * sin(u_time * 2.0) * 0.2;
    
    // Combine all warp effects
    theta += warp1 + warp2 + warp3 + momentum_warp;
    
    // Add radial distortion based on intensity
    r += sin(u_time * speed * 2.0) * u_intensity * 0.1;
    
    return vec2(cos(theta), sin(theta)) * r;
}

// Enhanced color generation with state-specific palettes
vec3 intelligentColor(vec2 p, float time) {
    vec3 color = vec3(0.0);
    
    // Add noise texture for organic feel
    float noise_val = fbm(p * 4.0 + time * 0.5);
    
    if (u_brain_state == STATE_CALM) {
        // CALM - serene blue gradients with gentle flow
        float wave1 = sin(p.x * 2.0 + time * 0.3) * 0.5 + 0.5;
        float wave2 = sin(p.y * 2.0 + time * 0.2) * 0.5 + 0.5;
        float wave3 = sin(length(p) * 3.0 + time * 0.25) * 0.5 + 0.5;
        
        color = vec3(
            wave1 * 0.3 + noise_val * 0.2,
            wave2 * 0.4 + noise_val * 0.3,
            wave3 * 0.8 + noise_val * 0.1
        );
    }
    else if (u_brain_state == STATE_FLOW) {
        // FLOW - cyan/green flowing patterns
        float flow1 = sin(p.x * 5.0 + p.y * 3.0 + time) * 0.5 + 0.5;
        float flow2 = sin(p.x * 3.0 - p.y * 5.0 + time * 1.2) * 0.5 + 0.5;
        float flow3 = sin(length(p) * 8.0 - time * 0.8) * 0.5 + 0.5;
        
        color = vec3(
            flow1 * 0.4 + noise_val * 0.3,
            flow2 * 0.7 + noise_val * 0.2,
            flow3 * 0.5 + noise_val * 0.3
        );
    }
    else if (u_brain_state == STATE_DRIVE) {
        // DRIVE - energetic orange/red radial patterns
        float r = length(p);
        float theta = atan(p.y, p.x);
        float drive1 = sin(r * 10.0 - time * 5.0 + theta) * 0.5 + 0.5;
        float drive2 = sin(r * 12.0 + time * 3.0 - theta) * 0.5 + 0.5;
        float drive3 = sin(r * 8.0 + time * 4.0) * 0.5 + 0.5;
        
        color = vec3(
            drive1 * 0.9 + noise_val * 0.1,
            drive2 * 0.4 + noise_val * 0.2,
            drive3 * 0.2 + noise_val * 0.1
        );
    }
    else if (u_brain_state == STATE_OVERDRIVE) {
        // OVERDRIVE - intense magenta/purple complex patterns
        float r = length(p);
        float theta = atan(p.y, p.x);
        float over1 = sin(r * 20.0 - time * 8.0 + theta * 2.0) * 0.5 + 0.5;
        float over2 = sin(r * 15.0 + time * 6.0 - theta * 3.0) * 0.5 + 0.5;
        float over3 = sin(r * 25.0 + time * 10.0 + theta) * 0.5 + 0.5;
        
        color = vec3(
            over1 * 0.8 + noise_val * 0.2,
            over2 * 0.3 + noise_val * 0.1,
            over3 * 0.9 + noise_val * 0.1
        );
    }
    else if (u_brain_state == STATE_CHAOS) {
        // CHAOS - explosive rainbow noise with structure
        float chaos1 = fract(sin(dot(p * time * 2.0, vec2(12.9898, 78.233))) * 43758.5453);
        float chaos2 = fract(sin(dot(p * time * 2.2, vec2(93.7434, 46.231))) * 23421.6312);
        float chaos3 = fract(sin(dot(p * time * 1.8, vec2(54.1234, 89.345))) * 12345.6789);
        
        color = vec3(
            chaos1 * 0.9 + noise_val * 0.1,
            chaos2 * 0.8 + noise_val * 0.2,
            chaos3 * 0.7 + noise_val * 0.3
        );
    }
    
    // Apply intensity and color enhancement
    color *= u_color_intensity;
    
    // Advanced color shifting with momentum
    float shift_amount = u_color_shift + u_momentum * 0.5;
    if (shift_amount > 0.0) {
        // Warm shift - towards red/orange
        color.r = mix(color.r, 1.0, shift_amount * 0.4);
        color.g = mix(color.g, color.r * 0.8, shift_amount * 0.3);
        color.b = mix(color.b, 0.0, shift_amount * 0.2);
    } else {
        // Cool shift - towards blue/cyan
        color.b = mix(color.b, 1.0, abs(shift_amount) * 0.4);
        color.g = mix(color.g, color.b * 0.8, abs(shift_amount) * 0.3);
        color.r = mix(color.r, 0.0, abs(shift_amount) * 0.2);
    }
    
    return color;
}

// Enhanced particle system with physics
vec3 advancedParticles(vec2 p, vec3 color, float time) {
    float particle_count = u_particle_density * 20.0;
    
    for(int i = 0; i < 20; i++) {
        if(i >= particle_count) break;
        
        // Particle physics with momentum
        float particle_phase = i * 0.618033988749895; // Golden ratio
        vec2 particle_base = vec2(
            sin(particle_phase + time * u_motion_speed) * 0.6,
            cos(particle_phase * 1.3 + time * u_motion_speed * 0.8) * 0.6
        );
        
        // Add brain state influence
        if(u_brain_state == STATE_CHAOS) {
            particle_base += vec2(
                sin(time * 10.0 + i) * 0.3,
                cos(time * 8.0 + i * 1.5) * 0.3
            );
        }
        
        // Momentum influence
        particle_base += u_momentum * vec2(sin(time * 3.0), cos(time * 2.5)) * 0.2;
        
        // Distance-based particle rendering
        float dist = length(p - particle_base);
        float particle_size = 0.03 + u_intensity * 0.02;
        
        if(dist < particle_size) {
            // Soft particle glow
            float glow = 1.0 - (dist / particle_size);
            glow = pow(glow, 3.0) * u_particle_activity;
            
            // Energy-based particle color
            vec3 particle_color;
            if(u_energy < 0.3) {
                particle_color = vec3(0.2, 0.4, 1.0); // Blue - low energy
            } else if(u_energy < 0.6) {
                particle_color = vec3(0.4, 1.0, 0.4); // Green - medium energy
            } else if(u_energy < 0.8) {
                particle_color = vec3(1.0, 0.8, 0.2); // Yellow - high energy
            } else {
                particle_color = vec3(1.0, 0.2, 0.4); // Red - maximum energy
            }
            
            // Add particle glow
            particle_color *= (1.0 + glow * 0.5);
            
            color = mix(color, particle_color, glow);
        }
    }
    
    return color;
}

// Enhanced glow with bloom simulation
vec3 advancedGlow(vec3 color, float intensity) {
    // Multiple glow layers for depth
    float glow1 = u_glow_amount * intensity * 0.5;
    float glow2 = u_glow_amount * intensity * 0.3;
    float glow3 = u_glow_amount * intensity * 0.2;
    
    // Add color-specific glow
    vec3 glow_color = color;
    glow_color.r += glow1 * 0.8;
    glow_color.g += glow2 * 0.6;
    glow_color.b += glow3 * 0.9;
    
    // Simulate bloom by adding brightened version
    vec3 bloom = color * 2.0;
    bloom = clamp(bloom, 0.0, 1.0);
    glow_color = mix(glow_color, bloom, u_glow_amount * 0.3);
    
    return glow_color;
}

// Enhanced feedback with motion blur
vec3 enhancedFeedback(vec2 uv, vec3 color) {
    // Motion-based feedback offset
    vec2 motion_offset = u_momentum * vec2(
        sin(u_time * 2.0) * 0.02,
        cos(u_time * 1.5) * 0.02
    );
    
    // Multiple feedback samples for motion blur
    vec2 feedback_uv = uv + motion_offset;
    vec3 feedback1 = texture(u_prev_frame, feedback_uv).rgb;
    vec3 feedback2 = texture(u_prev_frame, uv).rgb;
    vec3 feedback3 = texture(u_prev_frame, uv - motion_offset * 0.5).rgb;
    
    // Weighted feedback mix
    vec3 feedback = mix(feedback1, feedback2, 0.5);
    feedback = mix(feedback, feedback3, 0.3);
    
    // Intensity-based feedback strength
    float feedback_strength = 0.1 + u_intensity * 0.3 + u_stability * 0.1;
    
    return mix(color, feedback, feedback_strength);
}

void main() {
    // Calculate screen coordinates
    vec2 uv = gl_FragCoord.xy / u_resolution;
    vec2 p = uv * 2.0 - 1.0;
    
    // Apply advanced intelligence-driven warp
    p = advancedWarp(p, u_warp_strength, u_warp_speed);
    
    // Get enhanced intelligent color
    vec3 color = intelligentColor(p, u_brain_time);
    
    // Add advanced particle system
    color = advancedParticles(p, color, u_brain_time);
    
    // Apply enhanced glow
    color = advancedGlow(color, u_intensity);
    
    // Apply stability-based damping
    color *= mix(1.0, u_stability, 0.3);
    
    // Apply motion damping with subtle effect
    if(u_motion_damping > 0.0) {
        color = mix(color, vec3(0.5), u_motion_damping * 0.05);
    }
    
    // Enhanced feedback with motion blur
    color = enhancedFeedback(uv, color);
    
    // Final color correction and enhancement
    color = pow(color, vec3(0.9)); // Subtle gamma correction
    color = clamp(color, 0.0, 1.0);
    
    // Final output
    FragColor = vec4(color, 1.0);
}
