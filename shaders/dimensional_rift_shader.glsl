#version 330 core

// Dimensional Rift Shader - Spacetime Distortion Effects
uniform float u_time;
uniform vec2 u_resolution;
uniform sampler2D u_prev_frame;
uniform vec4 u_audio;
uniform float u_feedback;

in vec2 v_texcoord;
out vec4 FragColor;

// Dimensional parameters
#define MAX_RIFTS 4
#define RIFT_STRENGTH 0.3
#define QUANTUM_NOISE_SCALE 10.0
#define DIMENSIONAL_LAYERS 5

// Rift structure
struct DimensionalRift {
    vec2 position;
    float strength;
    float rotation;
    float frequency;
    vec3 color;
    int dimension_type;
};

// Random functions
float random(vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}

float noise(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

float fbm(vec2 p) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    
    for(int i = 0; i < 4; i++) {
        value += amplitude * noise(p * frequency);
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

// Generate dimensional rift
DimensionalRift generateRift(int index) {
    DimensionalRift rift;
    
    // Position influenced by audio
    float audio_intensity = u_audio.x + u_audio.y + u_audio.z;
    rift.position = vec2(
        sin(float(index) * 1.7 + u_time * 0.1) * 0.6,
        cos(float(index) * 2.3 + u_time * 0.15) * 0.6
    );
    
    // Strength based on audio
    rift.strength = RIFT_STRENGTH * (0.5 + audio_intensity);
    
    // Rotation
    rift.rotation = u_time * (0.5 + float(index) * 0.3);
    
    // Frequency for quantum effects
    rift.frequency = 2.0 + float(index) * 1.5 + audio_intensity * 2.0;
    
    // Color based on dimension type
    rift.dimension_type = index % 5;
    
    switch(rift.dimension_type) {
        case 0: // Time dimension
            rift.color = vec3(0.8, 0.2, 0.9); // Purple
            break;
        case 1: // Space dimension
            rift.color = vec3(0.2, 0.8, 0.9); // Cyan
            break;
        case 2: // Quantum dimension
            rift.color = vec3(0.9, 0.8, 0.2); // Yellow
            break;
        case 3: // Gravity dimension
            rift.color = vec3(0.9, 0.2, 0.3); // Red
            break;
        case 4: // Energy dimension
            rift.color = vec3(0.2, 0.9, 0.3); // Green
            break;
    }
    
    return rift;
}

// Calculate dimensional distortion
vec2 dimensionalDistortion(vec2 pos, DimensionalRift rift) {
    vec2 r = pos - rift.position;
    float distance = length(r);
    
    if (distance < 0.01) distance = 0.01; // Prevent singularity
    
    // Distortion based on rift type
    vec2 distortion = vec2(0.0);
    
    switch(rift.dimension_type) {
        case 0: // Time distortion - temporal stretching
            float time_warp = sin(distance * rift.frequency - u_time * 2.0) * rift.strength;
            distortion = r * time_warp;
            break;
            
        case 1: // Space distortion - spatial folding
            float space_fold = cos(distance * rift.frequency + rift.rotation) * rift.strength;
            vec2 fold_dir = normalize(r);
            distortion = fold_dir * space_fold;
            break;
            
        case 2: // Quantum distortion - probability field
            float quantum_noise = fbm(pos * QUANTUM_NOISE_SCALE + u_time * 0.5);
            vec2 quantum_dir = vec2(
                sin(quantum_noise * 10.0),
                cos(quantum_noise * 10.0)
            );
            distortion = quantum_dir * rift.strength * quantum_noise;
            break;
            
        case 3: // Gravity distortion - curvature
            float gravity_well = rift.strength / (distance * distance);
            distortion = normalize(r) * gravity_well * 0.1;
            break;
            
        case 4: // Energy distortion - wave interference
            float energy_wave = sin(distance * rift.frequency - u_time * 3.0) * 
                              cos(distance * rift.frequency * 0.7 + u_time * 2.0);
            distortion = vec2(energy_wave, energy_wave * 0.7) * rift.strength;
            break;
    }
    
    return distortion;
}

// Render dimensional rift core
vec3 renderRiftCore(vec2 pos, DimensionalRift rift) {
    vec2 r = pos - rift.position;
    float distance = length(r);
    
    // Core radius
    float core_radius = 0.05 + rift.strength * 0.1;
    
    if (distance < core_radius) {
        // Intense core
        float intensity = 1.0 - (distance / core_radius);
        intensity = pow(intensity, 3.0);
        
        // Pulsing effect
        float pulse = sin(u_time * rift.frequency) * 0.3 + 0.7;
        intensity *= pulse;
        
        // Color shifting based on dimension
        vec3 core_color = rift.color;
        
        // Add quantum noise for quantum rifts
        if (rift.dimension_type == 2) {
            float quantum = fbm(pos * 20.0 + u_time * 2.0);
            core_color = mix(core_color, hsv2rgb(vec3(quantum, 1.0, 1.0)), 0.5);
        }
        
        return core_color * intensity;
    }
    
    return vec3(0.0);
}

// Render dimensional energy field
vec3 renderEnergyField(vec2 pos, DimensionalRift rift) {
    vec2 r = pos - rift.position;
    float distance = length(r);
    
    // Field radius
    float field_radius = 0.3 + rift.strength * 0.2;
    
    if (distance < field_radius) {
        // Energy field intensity
        float field_intensity = 1.0 - (distance / field_radius);
        field_intensity = pow(field_intensity, 2.0);
        
        // Rotating field pattern
        float angle = atan(r.y, r.x);
        float field_pattern = sin(angle * 4.0 + rift.rotation + distance * rift.frequency);
        field_pattern = field_pattern * 0.5 + 0.5;
        
        // Combine with distance falloff
        field_intensity *= field_pattern;
        
        // Audio modulation
        float audio_mod = (u_audio.x + u_audio.y + u_audio.z) / 3.0;
        field_intensity *= (0.7 + audio_mod * 0.6);
        
        return rift.color * field_intensity * 0.5;
    }
    
    return vec3(0.0);
}

// Render dimensional particles
vec3 renderDimensionalParticles(vec2 pos, DimensionalRift rift, float time) {
    vec3 particle_color = vec3(0.0);
    float particle_count = 20.0;
    
    for(int i = 0; i < int(particle_count); i++) {
        // Particle orbit parameters
        float orbit_radius = 0.1 + (i / particle_count) * 0.2;
        float orbit_speed = rift.frequency * (1.0 + i * 0.1);
        
        // Calculate particle position
        float angle = time * orbit_speed + i * 0.618033988749895;
        vec2 particle_pos = rift.position + vec2(cos(angle), sin(angle)) * orbit_radius;
        
        // Add dimensional-specific motion
        switch(rift.dimension_type) {
            case 0: // Time particles - temporal drift
                particle_pos.x += sin(time * 5.0 + i) * 0.02;
                break;
            case 1: // Space particles - spatial jump
                if (sin(time * 3.0 + i) > 0.5) {
                    particle_pos += vec2(cos(i), sin(i)) * 0.05;
                }
                break;
            case 2: // Quantum particles - probability cloud
                float quantum = fbm(particle_pos * 10.0 + time);
                particle_pos += vec2(quantum - 0.5) * 0.03;
                break;
            case 3: // Gravity particles - spiral motion
                float spiral_angle = angle + time * 2.0;
                particle_pos += vec2(cos(spiral_angle), sin(spiral_angle)) * 0.01;
                break;
            case 4: // Energy particles - wave motion
                particle_pos.y += sin(time * 10.0 + i * 2.0) * 0.02;
                break;
        }
        
        // Check distance to particle
        float dist = length(pos - particle_pos);
        float particle_size = 0.008 + rift.strength * 0.005;
        
        if (dist < particle_size) {
            // Particle glow
            float glow = 1.0 - (dist / particle_size);
            glow = pow(glow, 4.0);
            
            // Particle color with energy-based brightness
            vec3 color = rift.color;
            color *= (1.0 + glow * 0.5);
            
            particle_color += color * glow;
        }
    }
    
    return particle_color;
}

// Apply dimensional distortion to texture
vec3 applyDistortionToTexture(vec2 uv, DimensionalRift[MAX_RIFTS] rifts, sampler2D tex) {
    vec2 distorted_uv = uv;
    
    // Calculate total distortion
    vec2 total_distortion = vec2(0.0);
    
    for(int i = 0; i < MAX_RIFTS; i++) {
        vec2 pos = uv * 2.0 - 1.0;
        vec2 distortion = dimensionalDistortion(pos, rifts[i]);
        total_distortion += distortion;
    }
    
    // Apply distortion to UV coordinates
    distorted_uv += total_distortion * 0.2;
    
    // Sample from distorted position
    vec3 color = texture(tex, distorted_uv).rgb;
    
    // Add dimensional color bleeding
    for(int i = 0; i < MAX_RIFTS; i++) {
        vec2 r = uv * 2.0 - 1.0 - rifts[i].position;
        float distance = length(r);
        
        if (distance < 0.4) {
            float bleed_intensity = 1.0 - (distance / 0.4);
            bleed_intensity = pow(bleed_intensity, 2.0);
            color += rifts[i].color * bleed_intensity * 0.1;
        }
    }
    
    return color;
}

void main() {
    vec2 uv = gl_FragCoord.xy / u_resolution;
    vec2 pos = uv * 2.0 - 1.0;
    
    // Generate dimensional rifts
    DimensionalRift[MAX_RIFTS] rifts;
    for(int i = 0; i < MAX_RIFTS; i++) {
        rifts[i] = generateRift(i);
    }
    
    vec3 color = vec3(0.0);
    
    // Start with distorted background
    color = applyDistortionToTexture(uv, rifts, u_prev_frame);
    
    // Add dimensional energy fields
    for(int i = 0; i < MAX_RIFTS; i++) {
        color += renderEnergyField(pos, rifts[i]);
    }
    
    // Render dimensional particles
    for(int i = 0; i < MAX_RIFTS; i++) {
        color += renderDimensionalParticles(pos, rifts[i], u_time);
    }
    
    // Render rift cores
    for(int i = 0; i < MAX_RIFTS; i++) {
        color += renderRiftCore(pos, rifts[i]);
    }
    
    // Add quantum noise overlay
    float quantum_field = fbm(pos * QUANTUM_NOISE_SCALE + u_time * 0.3);
    vec3 quantum_color = hsv2rgb(vec3(quantum_field, 0.3, 0.1));
    color += quantum_color;
    
    // Audio-reactive enhancement
    float audio_intensity = (u_audio.x + u_audio.y + u_audio.z) / 3.0;
    color *= (0.8 + audio_intensity * 0.4);
    
    // Feedback for continuity
    vec4 feedback = texture(u_prev_frame, uv);
    color = mix(color, feedback.rgb, u_feedback * 0.3);
    
    // Final color correction
    color = pow(color, vec3(0.9)); // Gamma correction
    color = clamp(color, 0.0, 1.0);
    
    FragColor = vec4(color, 1.0);
}
