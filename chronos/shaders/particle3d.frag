#version 460 core

/*
 * 3D Particle Fragment Shader
 * 
 * Renders particles as glowing spheres with species coloring.
 */

in float vEnergy;
in float vSpecies;
in vec3 vWorldPos;
in float vAggression;
in float vDefense;

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

// Predator-prey coloring: red (predator) <-> blue (prey) <-> green (balanced)
vec3 predatorPreyColor(float aggression, float defense, float energy) {
    // Predator score: high = predator (red), low = prey (blue), middle = balanced (green)
    float predatorScore = (aggression - defense + 1.0) * 0.5;  // Normalize to 0-1
    
    vec3 predatorColor = vec3(1.0, 0.15, 0.05);   // Aggressive red-orange
    vec3 preyColor = vec3(0.1, 0.4, 1.0);          // Defensive blue
    vec3 balancedColor = vec3(0.2, 0.95, 0.3);     // Neutral green
    
    vec3 color;
    if (predatorScore > 0.6) {
        // Predator range: interpolate green -> red
        color = mix(balancedColor, predatorColor, (predatorScore - 0.6) / 0.4);
    } else if (predatorScore < 0.4) {
        // Prey range: interpolate blue -> green
        color = mix(preyColor, balancedColor, predatorScore / 0.4);
    } else {
        // Balanced range
        color = balancedColor;
    }
    
    // Brightness scales with energy
    return color * (0.5 + energy * 0.5);
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
    
    // Color based on predator-prey status (aggression vs defense)
    vec3 color = predatorPreyColor(vAggression, vDefense, vEnergy);
    
    // Glow effect - predators glow brighter!
    float glowStrength = 0.5 + max(0.0, vAggression) * 1.5;  // Aggressive = brighter glow
    float glow = exp(-dist * 2.0) * vEnergy * glowStrength;
    color += vec3(1.0) * glow * 0.5;
    
    // Subtle pulsing - faster for predators
    float pulseSpeed = 3.0 + max(0.0, vAggression) * 4.0;
    float pulse = sin(u_Time * pulseSpeed + vSpecies * 2.0) * 0.1 + 0.9;
    color *= pulse;
    
    // Alpha based on distance from center
    float alpha = sphere * vEnergy;
    
    FragColor = vec4(color, alpha);
}
