#version 330 core

// Advanced Particle Physics Shader
uniform float u_time;
uniform vec2 u_resolution;
uniform sampler2D u_prev_frame;
uniform vec4 u_audio;
uniform float u_feedback;
out vec4 FragColor;

// Physics constants
#define GRAVITY 0.1
#define FRICTION 0.98
#define PARTICLE_LIFE 3.0
#define MAX_PARTICLES 100

// Particle structure (simulated in shader)
struct Particle {
    vec2 position;
    vec2 velocity;
    float life;
    float energy;
    int type;
};

// Random functions
float random(vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}

float randomRange(vec2 st, float minVal, float maxVal) {
    return mix(minVal, maxVal, random(st));
}

// HSV to RGB for energy-based coloring
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// Generate particle based on audio
Particle generateParticle(float index, vec2 uv) {
    Particle p;
    
    // Position based on audio and screen position
    vec2 audio_influence = vec2(u_audio.x, u_audio.y) * 2.0 - 1.0;
    p.position = uv + audio_influence * 0.3;
    
    // Velocity based on audio energy
    float energy = u_audio.x + u_audio.y + u_audio.z;
    p.velocity = vec2(
        randomRange(vec2(index, u_time), -1.0, 1.0) * energy,
        randomRange(vec2(index + 100.0, u_time), -1.0, 1.0) * energy
    );
    
    // Life based on beat
    p.life = PARTICLE_LIFE * (0.5 + u_audio.w * 0.5);
    
    // Energy level
    p.energy = energy / 3.0;
    
    // Particle type based on dominant frequency
    if (u_audio.x > u_audio.y && u_audio.x > u_audio.z) {
        p.type = 0; // Bass - heavy particles
    } else if (u_audio.y > u_audio.z) {
        p.type = 1; // Mid - medium particles
    } else {
        p.type = 2; // Treble - light particles
    }
    
    return p;
}

// Update particle physics
void updateParticle(inout Particle p, float dt) {
    // Apply gravity
    p.velocity.y -= GRAVITY * dt;
    
    // Apply audio forces
    vec2 audio_force = vec2(u_audio.x, u_audio.y) * 0.5;
    p.velocity += audio_force * dt;
    
    // Apply friction
    p.velocity *= FRICTION;
    
    // Update position
    p.position += p.velocity * dt;
    
    // Decay life
    p.life -= dt;
    
    // Beat pulse effect
    if (u_audio.w > 0.6) {
        float pulse = (u_audio.w - 0.6) * 2.5;
        p.velocity += vec2(cos(u_time * 10.0), sin(u_time * 10.0)) * pulse * 0.2;
    }
}

// Render particle with effects
vec3 renderParticle(Particle p, vec2 uv) {
    float dist = length(uv - p.position);
    
    // Size based on type and energy
    float size = 0.02;
    if (p.type == 0) size = 0.03; // Bass - larger
    else if (p.type == 1) size = 0.02; // Mid - medium
    else size = 0.01; // Treble - smaller
    
    size *= (0.5 + p.energy * 0.5);
    
    // Particle shape with soft edges
    float particle = 1.0 - smoothstep(0.0, size, dist);
    particle = pow(particle, 2.0); // Soft falloff
    
    // Color based on energy and type
    vec3 color;
    float hue;
    
    if (p.type == 0) {
        hue = 0.0; // Red for bass
    } else if (p.type == 1) {
        hue = 0.3; // Green for mid
    } else {
        hue = 0.6; // Blue for treble
    }
    
    // Shift hue based on energy
    hue += p.energy * 0.2;
    
    // Saturation and value based on life
    float sat = 0.8 * (p.life / PARTICLE_LIFE);
    float val = particle * (p.life / PARTICLE_LIFE);
    
    color = hsv2rgb(vec3(hue, sat, val));
    
    // Add glow for high energy
    if (p.energy > 0.7) {
        float glow = particle * 0.5;
        color += vec3(glow * 0.3, glow * 0.2, glow * 0.1);
    }
    
    return color;
}

void main() {
    vec2 uv = gl_FragCoord.xy / u_resolution;
    vec3 color = vec3(0.0);
    
    // Generate and update particles
    float particle_count = mix(20.0, 100.0, u_audio.x + u_audio.y + u_audio.z);
    
    for(int i = 0; i < MAX_PARTICLES; i++) {
        if(i >= particle_count) break;
        
        // Generate particle
        Particle p = generateParticle(i, uv);
        
        // Update physics
        updateParticle(p, 0.016); // ~60fps timestep
        
        // Only render if alive
        if(p.life > 0.0) {
            vec3 particle_color = renderParticle(p, uv);
            color += particle_color;
        }
    }
    
    // Add particle trails (motion blur effect)
    vec2 motion_uv = uv - vec2(0.01, 0.01);
    vec3 trails = texture(u_prev_frame, motion_uv).rgb * 0.3;
    color += trails;
    
    // Background gradient
    vec3 bg = mix(vec3(0.02, 0.01, 0.05), vec3(0.05, 0.02, 0.08), uv.y);
    color += bg * 0.2;
    
    // Audio-reactive ambient glow
    float ambient_glow = (u_audio.x + u_audio.y + u_audio.z) / 3.0;
    color += vec3(ambient_glow * 0.1, ambient_glow * 0.05, ambient_glow * 0.15);
    
    // Feedback for continuity
    vec4 feedback = texture(u_prev_frame, uv);
    color = mix(color, feedback.rgb, u_feedback * 0.4);
    
    // Final enhancements
    color = pow(color, vec3(0.8)); // Gamma correction
    color = clamp(color, 0.0, 1.0);
    
    FragColor = vec4(color, 1.0);
}
