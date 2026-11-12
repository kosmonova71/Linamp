#version 330 core

// Spacetime Distortion Shader - Dimensional Rift Effects
uniform float u_time;
uniform vec2 u_resolution;
uniform sampler2D u_prev_frame;
uniform vec4 u_audio;
uniform float u_feedback;

// Dimensional parameters
uniform float u_dimensional_strength;
uniform float u_rift_intensity;
uniform float u_quantum_fluctuation;
uniform float u_warp_factor;
uniform vec2 u_rift_center;

in vec2 v_texcoord;
out vec4 FragColor;

// Mathematical constants
#define PI 3.14159265359
#define TWO_PI 6.28318530718
#define PHI 1.618033988749895 // Golden ratio
#define PLANCK_LENGTH 0.001

// Quantum noise functions
float quantumNoise(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

float smoothQuantumNoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    
    float a = quantumNoise(i);
    float b = quantumNoise(i + vec2(1.0, 0.0));
    float c = quantumNoise(i + vec2(0.0, 1.0));
    float d = quantumNoise(i + vec2(1.0, 1.0));
    
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fractalQuantumNoise(vec2 p) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    
    for(int i = 0; i < 8; i++) {
        value += amplitude * smoothQuantumNoise(p * frequency);
        frequency *= PHI;
        amplitude *= 0.5;
    }
    return value;
}

// 4D noise for dimensional effects
float noise4D(vec4 p) {
    vec4 i = floor(p);
    vec4 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    
    float a = quantumNoise(i.xy + i.zw * 0.1);
    float b = quantumNoise(i.xy + vec2(1.0, 0.0) + i.zw * 0.1);
    float c = quantumNoise(i.xy + vec2(0.0, 1.0) + i.zw * 0.1);
    float d = quantumNoise(i.xy + vec2(1.0, 1.0) + i.zw * 0.1);
    
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y) * (1.0 - f.w) + f.w;
}

// Dimensional transformation matrix
mat3 dimensionalMatrix(float time, float strength) {
    float angle = time * 0.5;
    float twist = sin(time * 0.3) * strength;
    float shear = cos(time * 0.7) * strength * 0.5;
    
    return mat3(
        cos(angle) - shear, sin(angle) + twist, 0.0,
        -sin(angle) + twist, cos(angle) - shear, 0.0,
        shear * 0.3, twist * 0.3, 1.0
    );
}

// Spacetime curvature calculation
vec3 spacetimeCurvature(vec2 pos, vec2 rift, float strength) {
    vec2 toRift = rift - pos;
    float r = length(toRift);
    
    // Avoid singularity
    r = max(r, PLANCK_LENGTH);
    
    // Schwarzschild-like curvature
    float curvature = strength / (r * r);
    
    // Tidal forces
    vec2 tidal = toRift / (r * r * r) * strength * 0.1;
    
    // Frame dragging (Lense-Thirring effect)
    float frameDrag = sin(r * 10.0 - u_time * 5.0) * curvature * 0.1;
    
    return vec3(curvature, tidal.x, tidal.y + frameDrag);
}

// Dimensional rift rendering
vec3 renderDimensionalRift(vec2 pos, vec2 rift, float time) {
    vec2 toRift = pos - rift;
    float r = length(toRift);
    float theta = atan(toRift.y, toRift.x);
    
    // Rift core
    float coreRadius = 0.05 + sin(time * 2.0) * 0.02;
    if (r < coreRadius) {
        // Pure dimensional void
        return vec3(0.0);
    }
    
    // Event horizon layers
    float horizon1 = coreRadius * 2.0;
    float horizon2 = coreRadius * 4.0;
    float horizon3 = coreRadius * 8.0;
    
    vec3 riftColor = vec3(0.0);
    
    if (r < horizon1) {
        // Inner horizon - intense energy
        float intensity = 1.0 - (r - coreRadius) / (horizon1 - coreRadius);
        riftColor = vec3(
            intensity * 0.9,
            intensity * 0.3,
            intensity * 1.0
        );
    } else if (r < horizon2) {
        // Middle horizon - dimensional bleeding
        float intensity = 1.0 - (r - horizon1) / (horizon2 - horizon1);
        riftColor = vec3(
            intensity * 0.6,
            intensity * 0.8,
            intensity * 0.4
        );
    } else if (r < horizon3) {
        // Outer horizon - quantum fluctuations
        float intensity = 1.0 - (r - horizon2) / (horizon3 - horizon2);
        float quantum = fractalQuantumNoise(pos * 10.0 + time * 2.0);
        riftColor = vec3(
            intensity * quantum * 0.4,
            intensity * quantum * 0.6,
            intensity * quantum * 0.8
        );
    }
    
    // Spiral distortion
    float spiral = sin(theta * 3.0 + r * 5.0 - time * 3.0) * 0.5 + 0.5;
    riftColor *= spiral;
    
    return riftColor * u_rift_intensity;
}

// Quantum field visualization
vec3 renderQuantumField(vec2 pos, float time) {
    // 4D quantum noise
    vec4 p4d = vec4(pos * 5.0, time * 0.5, time * 0.3);
    float quantum = noise4D(p4d);
    
    // Quantum fluctuations
    float fluctuation = fractalQuantumNoise(pos * 8.0 + time) * u_quantum_fluctuation;
    
    // Field lines
    float field1 = sin(pos.x * 10.0 + time * 2.0) * 0.5 + 0.5;
    float field2 = cos(pos.y * 8.0 - time * 1.5) * 0.5 + 0.5;
    float field3 = sin(length(pos) * 15.0 + time * 3.0) * 0.5 + 0.5;
    
    // Quantum foam
    vec3 foamColor = vec3(
        quantum * fluctuation,
        field1 * fluctuation * 0.8,
        field2 * fluctuation * 0.6
    );
    
    // Add field lines
    foamColor += vec3(field3 * 0.2, field1 * 0.3, field2 * 0.4);
    
    return foamColor;
}

// Dimensional portal effect
vec3 renderDimensionalPortal(vec2 pos, vec2 rift, float time) {
    vec2 toRift = pos - rift;
    float r = length(toRift);
    float theta = atan(toRift.y, toRift.x);
    
    // Portal activation threshold
    float activation = sin(time * 0.5) * 0.5 + 0.5;
    float portalRadius = 0.15 + activation * 0.1;
    
    if (r < portalRadius) {
        // Inside portal - dimensional travel
        float depth = 1.0 - (r / portalRadius);
        
        // Rotating dimensional layers
        float layer1 = sin(theta * 4.0 + time * 5.0) * depth;
        float layer2 = cos(theta * 6.0 - time * 3.0) * depth;
        float layer3 = sin(theta * 8.0 + time * 7.0) * depth;
        
        // Dimensional colors
        vec3 portalColor = vec3(
            abs(layer1) * 0.8,
            abs(layer2) * 0.6,
            abs(layer3) * 1.0
        );
        
        // Add quantum noise
        portalColor += fractalQuantumNoise(pos * 20.0 + time * 10.0) * depth * 0.3;
        
        return portalColor * activation;
    }
    
    return vec3(0.0);
}

// Spacetime distortion effect
vec2 applySpacetimeDistortion(vec2 pos, vec2 rift, float strength) {
    vec2 toRift = rift - pos;
    float r = length(toRift);
    
    // Avoid singularity
    if (r < PLANCK_LENGTH) {
        return rift;
    }
    
    // Gravitational lensing
    float lensing = strength / (r * r);
    vec2 lensed = normalize(toRift) * lensing * 0.1;
    
    // Frame dragging
    float frameDrag = sin(r * 5.0 - u_time * 3.0) * strength * 0.05;
    vec2 dragged = vec2(-toRift.y, toRift.x) / r * frameDrag;
    
    // Time dilation effect (spatial manifestation)
    float timeDilation = 1.0 / sqrt(1.0 - 2.0 * strength / r);
    vec2 dilated = pos * (1.0 / timeDilation);
    
    return pos + lensed + dragged + (dilated - pos) * 0.1;
}

// Dimensional bleeding effect
vec3 dimensionalBleeding(vec2 pos, float time) {
    // Sample from multiple dimensional layers
    vec2 offset1 = vec2(sin(time * 2.0), cos(time * 1.5)) * 0.01;
    vec2 offset2 = vec2(cos(time * 3.0), sin(time * 2.5)) * 0.02;
    vec2 offset3 = vec2(sin(time * 1.0), cos(time * 4.0)) * 0.015;
    
    vec3 layer1 = texture(u_prev_frame, pos + offset1).rgb;
    vec3 layer2 = texture(u_prev_frame, pos + offset2).rgb;
    vec3 layer3 = texture(u_prev_frame, pos + offset3).rgb;
    
    // Mix with dimensional weights
    float weight1 = sin(time * 1.7) * 0.5 + 0.5;
    float weight2 = cos(time * 2.3) * 0.5 + 0.5;
    float weight3 = sin(time * 3.1) * 0.5 + 0.5;
    
    return (layer1 * weight1 + layer2 * weight2 + layer3 * weight3) / 3.0;
}

void main() {
    vec2 uv = gl_FragCoord.xy / u_resolution;
    vec2 pos = uv * 2.0 - 1.0;
    pos.x *= u_resolution.x / u_resolution.y; // Aspect ratio correction
    
    // Audio-reactive rift center
    vec2 audioOffset = vec2(u_audio.x, u_audio.y) * 0.3;
    vec2 rift = u_rift_center + audioOffset;
    
    // Apply spacetime distortion
    vec2 distortedPos = applySpacetimeDistortion(pos, rift, u_dimensional_strength);
    
    // Apply dimensional transformation
    mat3 dimMatrix = dimensionalMatrix(u_time, u_warp_factor);
    vec3 transformedPos = dimMatrix * vec3(distortedPos, 1.0);
    vec2 finalPos = transformedPos.xy;
    
    // Base color - quantum vacuum
    vec3 color = vec3(0.02, 0.01, 0.03);
    
    // Add quantum field
    vec3 quantumField = renderQuantumField(finalPos, u_time);
    color += quantumField * 0.3;
    
    // Render dimensional rift
    vec3 riftColor = renderDimensionalRift(finalPos, rift, u_time);
    color += riftColor;
    
    // Render dimensional portal
    vec3 portalColor = renderDimensionalPortal(finalPos, rift, u_time);
    color += portalColor;
    
    // Add dimensional bleeding
    vec3 bleeding = dimensionalBleeding(uv, u_time);
    color = mix(color, bleeding, 0.2);
    
    // Apply spacetime curvature effects
    vec3 curvature = spacetimeCurvature(finalPos, rift, u_dimensional_strength);
    color.r *= (1.0 + curvature.x * 0.5); // Redshift
    color.gb *= (1.0 - curvature.x * 0.2); // Blueshift suppression
    
    // Audio-reactive enhancements
    float audioIntensity = (u_audio.x + u_audio.y + u_audio.z) / 3.0;
    u_dimensional_strength = mix(0.5, 2.0, audioIntensity);
    u_rift_intensity = mix(0.3, 1.0, audioIntensity);
    u_quantum_fluctuation = mix(0.1, 0.5, audioIntensity);
    
    // Feedback for temporal continuity
    vec4 feedback = texture(u_prev_frame, uv);
    color = mix(color, feedback.rgb, u_feedback * 0.4);
    
    // Final color correction
    color = pow(color, vec3(0.9)); // Gamma correction
    color = clamp(color, 0.0, 1.0);
    
    FragColor = vec4(color, 1.0);
}
