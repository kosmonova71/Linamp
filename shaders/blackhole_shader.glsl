#version 330 core

// Black Hole Gravity Wells Shader
uniform float u_time;
uniform vec2 u_resolution;
uniform sampler2D u_prev_frame;
uniform vec4 u_audio;
uniform float u_feedback;

in vec2 v_texcoord;
out vec4 FragColor;

// Black hole parameters
#define MAX_BLACK_HOLES 3
#define GRAVITATIONAL_CONSTANT 1.0
#define EVENT_HORIZON 0.05
#define ACCRETION_DISK_RADIUS 0.3
#define SCHWARZSCHILD_RADIUS 0.02

// Black hole structure
struct BlackHole {
    vec2 position;
    float mass;
    float rotation;
    vec3 color;
};

// Random function for black hole generation
float random(vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}

// Generate black holes based on audio
BlackHole generateBlackHole(int index) {
    BlackHole bh;
    
    // Position influenced by audio
    float audio_influence = u_audio.x + u_audio.y + u_audio.z;
    bh.position = vec2(
        sin(float(index) * 2.3 + u_time * 0.1) * 0.5,
        cos(float(index) * 1.7 + u_time * 0.15) * 0.5
    );
    
    // Mass based on bass response
    bh.mass = 0.5 + u_audio.x * 2.0 + random(vec2(float(index), u_time)) * 0.5;
    
    // Rotation speed
    bh.rotation = u_time * (1.0 + float(index) * 0.3);
    
    // Color based on frequency dominance
    if (u_audio.x > u_audio.y && u_audio.x > u_audio.z) {
        bh.color = vec3(0.8, 0.2, 0.1); // Red for bass
    } else if (u_audio.y > u_audio.z) {
        bh.color = vec3(0.2, 0.8, 0.1); // Green for mid
    } else {
        bh.color = vec3(0.1, 0.2, 0.8); // Blue for treble
    }
    
    return bh;
}

// Calculate gravitational field strength
float gravitationalField(vec2 pos, BlackHole bh) {
    vec2 r = pos - bh.position;
    float distance = length(r);
    
    // Prevent singularity at center
    distance = max(distance, SCHWARZSCHILD_RADIUS);
    
    // Newtonian gravity with relativistic corrections
    float field = GRAVITATIONAL_CONSTANT * bh.mass / (distance * distance);
    
    // Add frame dragging effect near black hole
    if (distance < ACCRETION_DISK_RADIUS) {
        float frame_drag = bh.mass * 0.1 / (distance + 0.01);
        field += frame_drag;
    }
    
    return field;
}

// Calculate spacetime distortion
vec2 spacetimeDistortion(vec2 pos, BlackHole[MAX_BLACK_HOLES] black_holes) {
    vec2 distortion = vec2(0.0);
    
    for(int i = 0; i < MAX_BLACK_HOLES; i++) {
        vec2 r = pos - black_holes[i].position;
        float distance = length(r);
        
        // Prevent singularity
        distance = max(distance, SCHWARZSCHILD_RADIUS);
        
        // Schwarzschild metric distortion
        float metric_factor = 1.0 - (2.0 * GRAVITATIONAL_CONSTANT * black_holes[i].mass) / distance;
        metric_factor = max(metric_factor, 0.1); // Prevent negative values
        
        // Apply distortion
        if (distance > SCHWARZSCHILD_RADIUS) {
            vec2 direction = normalize(r);
            float distortion_strength = (1.0 - metric_factor) * 0.5;
            
            // Radial distortion
            distortion += direction * distortion_strength;
            
            // Tangential distortion (frame dragging)
            vec2 tangent = vec2(-direction.y, direction.x);
            float frame_drag_strength = black_holes[i].mass * 0.1 / (distance * distance);
            distortion += tangent * frame_drag_strength * sin(black_holes[i].rotation);
        }
    }
    
    return distortion;
}

// Render accretion disk
vec3 renderAccretionDisk(vec2 pos, BlackHole bh) {
    vec2 r = pos - bh.position;
    float distance = length(r);
    
    if (distance < SCHWARZSCHILD_RADIUS) {
        return vec3(0.0); // Inside event horizon
    }
    
    if (distance < ACCRETION_DISK_RADIUS) {
        // Accretion disk glow
        float disk_intensity = 1.0 - (distance - SCHWARZSCHILD_RADIUS) / (ACCRETION_DISK_RADIUS - SCHWARZSCHILD_RADIUS);
        disk_intensity = pow(disk_intensity, 2.0);
        
        // Doppler shifting due to rotation
        float angle = atan(r.y, r.x);
        float doppler = sin(angle - bh.rotation) * 0.5 + 0.5;
        
        // Temperature gradient (hotter near black hole)
        float temperature = 1.0 - (distance - SCHWARZSCHILD_RADIUS) / ACCRETION_DISK_RADIUS;
        
        vec3 disk_color = bh.color;
        disk_color *= temperature;
        disk_color *= disk_intensity;
        disk_color *= (0.5 + doppler * 0.5); // Doppler effect
        
        // Add blue shift for approaching side
        if (cos(angle - bh.rotation) > 0) {
            disk_color.b *= 1.2;
        }
        
        return disk_color;
    }
    
    return vec3(0.0);
}

// Render gravitational lensing
vec3 gravitationalLensing(vec2 pos, BlackHole[MAX_BLACK_HOLES] black_holes, sampler2D tex) {
    vec2 distorted_pos = pos;
    
    // Calculate total distortion
    vec2 total_distortion = spacetimeDistortion(pos, black_holes);
    distorted_pos += total_distortion * 0.3;
    
    // Sample from distorted position
    vec3 color = texture(tex, distorted_pos).rgb;
    
    // Add lensing brightening effect
    for(int i = 0; i < MAX_BLACK_HOLES; i++) {
        vec2 r = pos - black_holes[i].position;
        float distance = length(r);
        
        if (distance > SCHWARZSCHILD_RADIUS && distance < ACCRETION_DISK_RADIUS * 2.0) {
            // Einstein ring effect
            float ring_intensity = 1.0 - (distance - SCHWARZSCHILD_RADIUS) / (ACCRETION_DISK_RADIUS * 2.0 - SCHWARZSCHILD_RADIUS);
            color += black_holes[i].color * ring_intensity * 0.3;
        }
    }
    
    return color;
}

// Orbiting particles
vec3 renderOrbitingParticles(vec2 pos, BlackHole[MAX_BLACK_HOLES] black_holes, float time) {
    vec3 particle_color = vec3(0.0);
    float particle_count = 50.0;
    
    for(int i = 0; i < int(particle_count); i++) {
        // Particle orbit parameters
        float orbit_index = i / particle_count;
        float orbit_radius = 0.2 + orbit_index * 0.4;
        float orbit_speed = 1.0 / (orbit_radius * orbit_radius); // Kepler's third law
        
        // Find dominant black hole for this orbit
        int dominant_hole = 0;
        float max_influence = 0.0;
        
        for(int j = 0; j < MAX_BLACK_HOLES; j++) {
            float influence = black_holes[j].mass / (orbit_radius * orbit_radius);
            if (influence > max_influence) {
                max_influence = influence;
                dominant_hole = j;
            }
        }
        
        // Calculate particle position
        float angle = time * orbit_speed + i * 0.618033988749895; // Golden ratio
        vec2 particle_pos = black_holes[dominant_hole].position + 
                          vec2(cos(angle), sin(angle)) * orbit_radius;
        
        // Add elliptical orbit perturbation
        float eccentricity = 0.2 * sin(i * 0.3);
        particle_pos += vec2(cos(angle * 2.0), sin(angle * 2.0)) * eccentricity;
        
        // Check distance to particle
        float dist = length(pos - particle_pos);
        float particle_size = 0.005 + u_audio.w * 0.01;
        
        if (dist < particle_size) {
            // Particle glow
            float glow = 1.0 - (dist / particle_size);
            glow = pow(glow, 3.0);
            
            // Color based on orbit radius (temperature)
            vec3 color = mix(
                vec3(0.2, 0.4, 1.0), // Cold - outer orbit
                vec3(1.0, 0.4, 0.2), // Hot - inner orbit
                1.0 - orbit_index
            );
            
            particle_color += color * glow;
        }
    }
    
    return particle_color;
}

void main() {
    vec2 uv = gl_FragCoord.xy / u_resolution;
    vec2 pos = uv * 2.0 - 1.0;
    
    // Generate black holes
    BlackHole[MAX_BLACK_HOLES] black_holes;
    for(int i = 0; i < MAX_BLACK_HOLES; i++) {
        black_holes[i] = generateBlackHole(i);
    }
    
    vec3 color = vec3(0.0);
    
    // Start with background
    color = vec3(0.02, 0.01, 0.05);
    
    // Add spacetime grid distortion
    vec2 grid_pos = pos * 10.0;
    vec2 distorted_grid = grid_pos + spacetimeDistortion(pos, black_holes) * 2.0;
    vec2 grid_fract = fract(distorted_grid);
    vec2 grid_step = smoothstep(0.0, 0.1, grid_fract) - smoothstep(0.9, 1.0, grid_fract);
    float grid_intensity = grid_step.x * grid_step.y;
    color += vec3(0.1, 0.05, 0.15) * grid_intensity * 0.3;
    
    // Render accretion disks
    for(int i = 0; i < MAX_BLACK_HOLES; i++) {
        color += renderAccretionDisk(pos, black_holes[i]);
    }
    
    // Render orbiting particles
    color += renderOrbitingParticles(pos, black_holes, u_time);
    
    // Apply gravitational lensing to previous frame
    vec3 lensed_background = gravitationalLensing(uv, black_holes, u_prev_frame);
    color = mix(color, lensed_background, 0.3);
    
    // Add event horizon effect
    for(int i = 0; i < MAX_BLACK_HOLES; i++) {
        vec2 r = pos - black_holes[i].position;
        float distance = length(r);
        
        if (distance < EVENT_HORIZON) {
            // Complete darkness inside event horizon
            color = vec3(0.0);
        } else if (distance < EVENT_HORIZON * 2.0) {
            // Gravitational redshift near event horizon
            float redshift = 1.0 - (distance - EVENT_HORIZON) / EVENT_HORIZON;
            color *= (1.0 - redshift * 0.8);
        }
    }
    
    // Audio-reactive enhancement
    float audio_intensity = (u_audio.x + u_audio.y + u_audio.z) / 3.0;
    color *= (0.8 + audio_intensity * 0.4);
    
    // Feedback for continuity
    vec4 feedback = texture(u_prev_frame, uv);
    color = mix(color, feedback.rgb, u_feedback * 0.2);
    
    // Final color correction
    color = pow(color, vec3(0.8)); // Gamma correction
    color = clamp(color, 0.0, 1.0);
    
    FragColor = vec4(color, 1.0);
}
