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

// HSL to RGB
vec3 hsl2rgb(vec3 c) {
    vec3 rgb = clamp(abs(mod(c.x * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
    return c.z + c.y * (rgb - 0.5) * (1.0 - abs(2.0 * c.z - 1.0));
}

vec3 speciesColor(float species, float energy) {
    float hue = mod(species * 0.3, 1.0);
    return hsl2rgb(vec3(hue, 0.7 + energy * 0.3, 0.4 + energy * 0.4));
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
    
    // Color based on species and energy
    vec3 color = speciesColor(vSpecies, vEnergy);
    
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
