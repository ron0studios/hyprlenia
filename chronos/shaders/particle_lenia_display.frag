#version 450 core

/*
 * PARTICLE LENIA - Display Shader
 * 
 * Renders the particle field with:
 * - Particle visualization (dots)
 * - Field visualization (U, R, G, E)
 * - Species-based coloring
 */

in vec2 TexCoord;
out vec4 FragColor;

// Particle structure (12 floats)
layout(std430, binding = 0) readonly buffer Particles {
    float particles[];
};

// Uniforms
uniform int u_NumParticles;
uniform float u_WorldWidth;
uniform float u_WorldHeight;
uniform float u_TranslateX;
uniform float u_TranslateY;
uniform float u_Zoom;
uniform float u_WindowWidth;
uniform float u_WindowHeight;

uniform float u_Wk;
uniform float u_MuK;
uniform float u_SigmaK2;
uniform float u_MuG;
uniform float u_SigmaG2;

uniform bool u_ShowFields;
uniform int u_FieldType;

// Colors
const vec3 BACKGROUND = vec3(0.005, 0.02, 0.05);
const vec3 FIELD_COLOR_1 = vec3(0.1, 0.3, 0.6);  // Blue for density
const vec3 FIELD_COLOR_2 = vec3(0.0, 0.8, 0.4);  // Green for growth

// HSL to RGB conversion
vec3 hsl2rgb(vec3 c) {
    vec3 rgb = clamp(abs(mod(c.x * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
    return c.z + c.y * (rgb - 0.5) * (1.0 - abs(2.0 * c.z - 1.0));
}

// Get species color
vec3 speciesColor(float species, float energy) {
    float hue = mod(species * 0.3, 1.0);
    float saturation = 0.7 + energy * 0.3;
    float lightness = 0.3 + energy * 0.4;
    return hsl2rgb(vec3(hue, saturation, lightness));
}

// Kernel function
float K(float r) {
    float diff = r - u_MuK;
    return u_Wk * exp(-diff * diff / u_SigmaK2);
}

// Growth function
float G(float u) {
    float diff = u - u_MuG;
    return exp(-diff * diff / u_SigmaG2);
}

// Read particle position and energy
vec4 readParticle(int idx) {
    int base = idx * 12;
    return vec4(
        particles[base + 0],  // x
        particles[base + 1],  // y
        particles[base + 4],  // energy
        particles[base + 5]   // species
    );
}

// Compute distance with wrapping
float wrappedDistance(vec2 pos1, vec2 pos2) {
    vec2 delta = pos2 - pos1;
    
    if (delta.x > u_WorldWidth) delta.x -= 2.0 * u_WorldWidth;
    if (delta.x < -u_WorldWidth) delta.x += 2.0 * u_WorldWidth;
    if (delta.y > u_WorldHeight) delta.y -= 2.0 * u_WorldHeight;
    if (delta.y < -u_WorldHeight) delta.y += 2.0 * u_WorldHeight;
    
    return length(delta);
}

// Calculate fields at position
vec4 computeFields(vec2 position) {
    float u = 0.0;
    float r = 0.0;
    
    for (int i = 0; i < u_NumParticles; i++) {
        vec4 p = readParticle(i);
        
        if (p.z < 0.01) continue; // Skip dead
        
        float dist = wrappedDistance(position, p.xy);
        
        // Density
        u += K(dist) * p.z;
        
        // Repulsion
        if (dist > 0.0001 && dist < 2.0) {
            float rep = max(1.0 - dist, 0.0);
            r += 0.5 * rep * rep;
        }
    }
    
    float g = G(u);
    float e = r - g;
    
    return vec4(u, r, g, e);
}

void main() {
    // Convert screen coords to world coords
    float aspect = u_WindowWidth / u_WindowHeight;
    vec2 uv = TexCoord * 2.0 - 1.0;
    
    vec2 worldPos;
    worldPos.x = uv.x * u_WorldWidth / u_Zoom + u_TranslateX;
    worldPos.y = uv.y * u_WorldHeight / u_Zoom + u_TranslateY;
    
    vec3 color = BACKGROUND;
    
    // Show field if enabled
    if (u_ShowFields && u_FieldType > 0) {
        vec4 fields = computeFields(worldPos);
        
        float fieldValue = 0.0;
        vec3 fieldColor = FIELD_COLOR_1;
        
        switch (u_FieldType) {
            case 1: // U - density
                fieldValue = min(fields.x * 2.0, 1.0);
                fieldColor = vec3(0.0, 0.3, 0.8);
                break;
            case 2: // R - repulsion
                fieldValue = min(fields.y, 1.0);
                fieldColor = vec3(0.8, 0.2, 0.0);
                break;
            case 3: // G - growth
                fieldValue = fields.z;
                fieldColor = vec3(0.0, 0.8, 0.3);
                break;
            case 4: // E - energy
                float e = fields.w;
                if (e > 0.0) {
                    fieldValue = min(e, 1.0);
                    fieldColor = vec3(0.8, 0.0, 0.0);
                } else {
                    fieldValue = min(-e, 1.0);
                    fieldColor = vec3(0.0, 0.8, 0.3);
                }
                break;
        }
        
        color = mix(BACKGROUND, fieldColor, fieldValue * 0.5);
    }
    
    // Draw particles
    float minDist = 1000.0;
    vec3 closestColor = vec3(1.0);
    float closestEnergy = 0.0;
    
    for (int i = 0; i < u_NumParticles; i++) {
        vec4 p = readParticle(i);
        
        if (p.z < 0.01) continue; // Skip dead
        
        float dist = wrappedDistance(worldPos, p.xy);
        
        if (dist < minDist) {
            minDist = dist;
            closestEnergy = p.z;
            closestColor = speciesColor(p.w, p.z);
        }
    }
    
    // Particle glow
    float particleRadius = 0.15 / u_Zoom;
    float glowRadius = 0.5 / u_Zoom;
    
    if (minDist < glowRadius) {
        float glow = 1.0 - minDist / glowRadius;
        glow = pow(glow, 2.0);
        color = mix(color, closestColor * 0.5, glow * closestEnergy * 0.5);
    }
    
    if (minDist < particleRadius) {
        float core = 1.0 - minDist / particleRadius;
        core = pow(core, 0.5);
        color = mix(color, closestColor, core);
    }
    
    // Very bright center
    if (minDist < particleRadius * 0.3) {
        color = mix(color, vec3(1.0), 0.8);
    }
    
    FragColor = vec4(color, 1.0);
}
