#version 330 core

// Orbiting Particle Fields Shader
uniform float u_time;
uniform vec2 u_resolution;
uniform sampler2D u_prev_frame;
uniform vec4 u_audio;
uniform float u_feedback;

in vec2 v_texcoord;
out vec4 FragColor;

// Particle system parameters
#define MAX_GRAVITY_WELLS 4
#define MAX_PARTICLES 200
#define ORBITAL_VELOCITY_FACTOR 0.8
#define PARTICLE_TRAIL_LENGTH 10

// Gravity well structure
struct GravityWell {
    vec2 position;
    float mass;
    float rotation_speed;
    vec3 color;
    int well_type;
};

// Particle structure
struct Particle {
    vec2 position;
    vec2 velocity;
    float mass;
    vec3 color;
    float life;
    int orbiting_well;
};

// Random functions
float random(vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}

float randomRange(vec2 st, float minVal, float maxVal) {
    return mix(minVal, maxVal, random(st));
}

// HSV to RGB conversion
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// Generate gravity wells based on audio
GravityWell generateGravityWell(int index) {
    GravityWell well;
    
    // Position influenced by audio
    float audio_intensity = u_audio.x + u_audio.y + u_audio.z;
    well.position = vec2(
        sin(float(index) * 1.7 + u_time * 0.05) * 0.5,
        cos(float(index) * 2.3 + u_time * 0.08) * 0.5
    );
    
    // Mass based on bass response
    well.mass = 0.5 + u_audio.x * 2.0 + random(vec2(float(index), u_time)) * 0.5;
    
    // Rotation speed
    well.rotation_speed = 0.5 + float(index) * 0.2 + audio_intensity * 0.5;
    
    // Type and color based on frequency
    if (u_audio.x > u_audio.y && u_audio.x > u_audio.z) {
        well.well_type = 0; // Bass well - red
        well.color = vec3(0.9, 0.3, 0.2);
    } else if (u_audio.y > u_audio.z) {
        well.well_type = 1; // Mid well - green
        well.color = vec3(0.3, 0.9, 0.2);
    } else {
        well.well_type = 2; // Treble well - blue
        well.color = vec3(0.2, 0.3, 0.9);
    }
    
    return well;
}

// Calculate gravitational force
vec2 calculateGravity(vec2 particle_pos, GravityWell well) {
    vec2 r = well.position - particle_pos;
    float distance = length(r);
    
    // Prevent singularity
    distance = max(distance, 0.05);
    
    // Newton's law of gravitation
    float force_magnitude = well.mass / (distance * distance);
    
    // Add relativistic corrections near massive objects
    if (distance < 0.2) {
        force_magnitude *= (1.0 + well.mass * 0.5);
    }
    
    return normalize(r) * force_magnitude;
}

// Generate particle
Particle generateParticle(int index, GravityWell[MAX_GRAVITY_WELLS] wells) {
    Particle p;
    
    // Random initial position
    p.position = vec2(
        randomRange(vec2(float(index), 0.0), -1.0, 1.0),
        randomRange(vec2(float(index), 1.0), -1.0, 1.0)
    );
    
    // Initial velocity based on nearest well
    float min_distance = 1000.0;
    int nearest_well = 0;
    
    for(int i = 0; i < MAX_GRAVITY_WELLS; i++) {
        float dist = length(p.position - wells[i].position);
        if (dist < min_distance) {
            min_distance = dist;
            nearest_well = i;
        }
    }
    
    // Set orbital velocity
    vec2 to_well = wells[nearest_well].position - p.position;
    float orbital_speed = ORBITAL_VELOCITY_FACTOR * sqrt(wells[nearest_well].mass / min_distance);
    vec2 orbital_direction = vec2(-to_well.y, to_well.x);
    p.velocity = normalize(orbital_direction) * orbital_speed;
    
    // Particle mass
    p.mass = randomRange(vec2(float(index), 2.0), 0.1, 0.5);
    
    // Color based on mass and well type
    float hue = mix(0.0, 0.8, p.mass); // Red to blue based on mass
    p.color = hsv2rgb(vec3(hue, 0.8, 1.0));
    
    // Life
    p.life = 1.0;
    
    // Orbiting well
    p.orbiting_well = nearest_well;
    
    return p;
}

// Update particle physics
void updateParticle(inout Particle p, GravityWell[MAX_GRAVITY_WELLS] wells, float dt) {
    // Calculate total gravitational force
    vec2 total_force = vec2(0.0);
    
    for(int i = 0; i < MAX_GRAVITY_WELLS; i++) {
        vec2 force = calculateGravity(p.position, wells[i]);
        total_force += force;
        
        // Check if particle is captured by this well
        float dist_to_well = length(p.position - wells[i].position);
        if (dist_to_well < 0.1) {
            p.orbiting_well = i;
        }
    }
    
    // Add audio-reactive forces
    vec2 audio_force = vec2(
        u_audio.x - u_audio.y,
        u_audio.y - u_audio.z
    ) * 0.2;
    total_force += audio_force;
    
    // Update velocity (F = ma, so a = F/m)
    p.velocity += (total_force / p.mass) * dt;
    
    // Add damping to prevent runaway velocities
    p.velocity *= 0.99;
    
    // Update position
    p.position += p.velocity * dt;
    
    // Decay life
    p.life -= dt * 0.1;
    
    // Respawn if dead
    if (p.life <= 0.0) {
        p.position = vec2(
            randomRange(vec2(p.position.x, u_time), -1.0, 1.0),
            randomRange(vec2(p.position.y, u_time), -1.0, 1.0)
        );
        p.life = 1.0;
        p.velocity *= 0.5; // Reset velocity
    }
    
    // Boundary check
    if (abs(p.position.x) > 1.2 || abs(p.position.y) > 1.2) {
        p.position = -p.position * 0.8; // Wrap around
    }
}

// Render particle with trail
vec3 renderParticle(Particle p, vec2 uv, float time) {
    vec2 r = uv - p.position;
    float distance = length(r);
    
    // Particle size based on mass and life
    float particle_size = 0.005 + p.mass * 0.01 * p.life;
    
    // Soft particle rendering
    float particle = 1.0 - smoothstep(0.0, particle_size, distance);
    particle = pow(particle, 2.0);
    
    // Color with life-based fading
    vec3 color = p.color * p.life;
    
    // Add glow for massive particles
    if (p.mass > 0.3) {
        float glow_size = particle_size * 3.0;
        float glow = 1.0 - smoothstep(0.0, glow_size, distance);
        glow = pow(glow, 1.5) * 0.3;
        color += p.color * glow;
    }
    
    return color * particle;
}

// Render particle trail
vec3 renderParticleTrail(Particle p, vec2 uv, float time) {
    vec3 trail_color = vec3(0.0);
    
    // Simulate trail positions
    for(int i = 0; i < PARTICLE_TRAIL_LENGTH; i++) {
        float trail_time = time - float(i) * 0.01;
        vec2 trail_pos = p.position - p.velocity * float(i) * 0.01;
        
        vec2 r = uv - trail_pos;
        float distance = length(r);
        
        float trail_size = 0.002 + p.mass * 0.005;
        float trail_intensity = 1.0 - float(i) / float(PARTICLE_TRAIL_LENGTH);
        trail_intensity *= p.life;
        
        float trail = 1.0 - smoothstep(0.0, trail_size, distance);
        trail = pow(trail, 3.0);
        
        trail_color += p.color * trail * trail_intensity * 0.3;
    }
    
    return trail_color;
}

// Render gravity well
vec3 renderGravityWell(GravityWell well, vec2 uv, float time) {
    vec2 r = uv - well.position;
    float distance = length(r);
    
    // Well core
    float core_radius = 0.02 + well.mass * 0.01;
    if (distance < core_radius) {
        float intensity = 1.0 - (distance / core_radius);
        intensity = pow(intensity, 3.0);
        
        // Pulsing effect
        float pulse = sin(time * well.rotation_speed * 2.0) * 0.2 + 0.8;
        intensity *= pulse;
        
        return well.color * intensity;
    }
    
    // Gravitational field visualization
    if (distance < 0.3) {
        float field_intensity = 1.0 - (distance / 0.3);
        field_intensity = pow(field_intensity, 2.0);
        
        // Rotating field lines
        float angle = atan(r.y, r.x);
        float field_pattern = sin(angle * 6.0 + time * well.rotation_speed);
        field_pattern = field_pattern * 0.5 + 0.5;
        
        field_intensity *= field_pattern * 0.2;
        
        return well.color * field_intensity;
    }
    
    return vec3(0.0);
}

// Render orbital paths
vec3 renderOrbitalPaths(GravityWell[MAX_GRAVITY_WELLS] wells, vec2 uv, float time) {
    vec3 path_color = vec3(0.0);
    
    for(int i = 0; i < MAX_GRAVITY_WELLS; i++) {
        // Calculate orbital path radius
        float orbital_radius = 0.15 + wells[i].mass * 0.1;
        
        // Draw orbital circle
        vec2 r = uv - wells[i].position;
        float distance = abs(length(r) - orbital_radius);
        
        if (distance < 0.005) {
            float intensity = 1.0 - (distance / 0.005);
            intensity = pow(intensity, 4.0);
            
            // Animated dashed line
            float angle = atan(r.y, r.x);
            float dash = sin(angle * 20.0 - time * wells[i].rotation_speed) * 0.5 + 0.5;
            
            path_color += wells[i].color * intensity * dash * 0.3;
        }
    }
    
    return path_color;
}

void main() {
    vec2 uv = gl_FragCoord.xy / u_resolution;
    vec2 pos = uv * 2.0 - 1.0;
    
    // Generate gravity wells
    GravityWell[MAX_GRAVITY_WELLS] wells;
    for(int i = 0; i < MAX_GRAVITY_WELLS; i++) {
        wells[i] = generateGravityWell(i);
    }
    
    vec3 color = vec3(0.0);
    
    // Background
    color = vec3(0.01, 0.005, 0.02);
    
    // Render orbital paths
    color += renderOrbitalPaths(wells, pos, u_time);
    
    // Render gravity wells
    for(int i = 0; i < MAX_GRAVITY_WELLS; i++) {
        color += renderGravityWell(wells[i], pos, u_time);
    }
    
    // Generate and render particles
    float particle_count = mix(50.0, MAX_PARTICLES, (u_audio.x + u_audio.y + u_audio.z) / 3.0);
    
    for(int i = 0; i < MAX_PARTICLES; i++) {
        if(i >= particle_count) break;
        
        Particle p = generateParticle(int(i), wells);
        updateParticle(p, wells, 0.016); // ~60fps timestep
        
        // Render particle trail
        color += renderParticleTrail(p, pos, u_time);
        
        // Render particle
        color += renderParticle(p, pos, u_time);
    }
    
    // Add audio-reactive glow
    float audio_intensity = (u_audio.x + u_audio.y + u_audio.z) / 3.0;
    vec3 audio_glow = vec3(
        u_audio.x * 0.3,
        u_audio.y * 0.2,
        u_audio.z * 0.4
    );
    color += audio_glow * audio_intensity;
    
    // Feedback for continuity
    vec4 feedback = texture(u_prev_frame, uv);
    color = mix(color, feedback.rgb, u_feedback * 0.2);
    
    // Final color correction
    color = pow(color, vec3(0.8)); // Gamma correction
    color = clamp(color, 0.0, 1.0);
    
    FragColor = vec4(color, 1.0);
}
