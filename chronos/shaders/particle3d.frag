#version 460 core

/*
 * 3D Particle Fragment Shader
 * 
 * Renders particles as glowing spheres with species coloring.
 */

in float vEnergy;
in float vSpecies;
in vec3 vWorldPos;

out vec4 FragColor;

uniform vec3 u_CameraPos;
uniform float u_Time;

// Channel-based coloring for multi-channel Lenia
vec3 channelColor(float channel, float energy) {
    // Channel 0: Blue/Cyan spectrum
    // Channel 1: Orange/Red spectrum
    vec3 baseColor;
    if (channel < 0.5) {
        // Channel 0: Cool blue
        baseColor = vec3(0.2, 0.5, 1.0);
    } else {
        // Channel 1: Warm orange
        baseColor = vec3(1.0, 0.4, 0.15);
    }
    // Brighten based on energy
    return baseColor * (0.5 + energy * 0.5);
}

void main() {
    // Skip dead particles
    if (vEnergy < 0.01) {
        discard;
    }
    
    // Create circular point sprite
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float dist = length(coord);
    
    if (dist > 1.0) {
        discard;
    }
    
    // Soft sphere effect
    float sphere = 1.0 - dist;
    sphere = pow(sphere, 1.5);
    
    // Color based on channel and energy
    vec3 color = channelColor(vSpecies, vEnergy);
    
    // Glow effect
    float glow = exp(-dist * 2.0) * vEnergy;
    color += vec3(1.0) * glow * 0.5;
    
    // Subtle pulsing
    float pulse = sin(u_Time * 3.0 + vSpecies * 2.0) * 0.1 + 0.9;
    color *= pulse;
    
    // Alpha based on distance from center
    float alpha = sphere * vEnergy;
    
    FragColor = vec4(color, alpha);
}
