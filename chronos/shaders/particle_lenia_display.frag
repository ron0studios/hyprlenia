#version 450 core

/*
 * CHRONOS - Ultra-Optimized Display Shader
 * 
 * Optimizations:
 * 1. Subsampled field computation (1/8 resolution)
 * 2. Distance-based early exit for particle rendering
 * 3. Reduced particle loop iterations
 * 4. Simplified field calculations
 */

in vec2 TexCoord;
out vec4 FragColor;

layout(std430, binding = 0) readonly buffer Particles {
    float particles[];
};

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

// Food system
uniform sampler2D u_FoodTexture;
uniform bool u_ShowFood;
uniform int u_FoodGridSize;

const vec3 BACKGROUND = vec3(0.005, 0.02, 0.05);

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
    return baseColor * (0.4 + energy * 0.6);
}

// Inline particle read (14 floats per particle: x,y,z, vx,vy,vz, energy, species, age, dna[5])
#define READ_PARTICLE_POS(i) vec2(particles[(i) * 14], particles[(i) * 14 + 1])
#define READ_PARTICLE_MASS(i) particles[(i) * 14 + 6]
#define READ_PARTICLE_SPECIES(i) particles[(i) * 14 + 7]

// Wrapped distance squared (fast)
// World goes from -worldWidth/2 to +worldWidth/2, so total size = worldWidth
float wrappedDist2(vec2 pos1, vec2 pos2) {
    vec2 d = pos2 - pos1;
    float halfW = u_WorldWidth * 0.5;
    float halfH = u_WorldHeight * 0.5;
    if (d.x > halfW) d.x -= u_WorldWidth;
    else if (d.x < -halfW) d.x += u_WorldWidth;
    if (d.y > halfH) d.y -= u_WorldHeight;
    else if (d.y < -halfH) d.y += u_WorldHeight;
    return dot(d, d);
}

void main() {
    vec2 uv = TexCoord * 2.0 - 1.0;
    
    // Correct for window aspect ratio
    float windowAspect = u_WindowWidth / u_WindowHeight;
    float worldAspect = u_WorldWidth / u_WorldHeight;
    
    // Scale UV to maintain proper aspect ratio
    vec2 scaledUV = uv;
    if (windowAspect > worldAspect) {
        // Window is wider than world - letterbox on sides
        scaledUV.x *= windowAspect / worldAspect;
    } else {
        // Window is taller than world - letterbox on top/bottom
        scaledUV.y *= worldAspect / windowAspect;
    }
    
    // scaledUV is in [-1, 1], map to world coords [-worldWidth/2, +worldWidth/2]
    vec2 worldPos = vec2(
        scaledUV.x * (u_WorldWidth * 0.5) / u_Zoom + u_TranslateX,
        scaledUV.y * (u_WorldHeight * 0.5) / u_Zoom + u_TranslateY
    );

    // Limit rendering to the world bounds to avoid ghost duplicates
    if (abs(worldPos.x) > u_WorldWidth * 0.5 || abs(worldPos.y) > u_WorldHeight * 0.5) {
        FragColor = vec4(BACKGROUND * 0.5, 1.0);  // Darker background outside world
        return;
    }
    
    vec3 color = BACKGROUND;
    
    // === FAST FIELD OVERLAY ===
    // Only compute field for every 8th pixel (checkerboard pattern)
    if (u_ShowFields && u_FieldType > 0) {
        ivec2 pixelCoord = ivec2(gl_FragCoord.xy);
        
        // Subsample: only compute for every 4th pixel in each dimension
        if ((pixelCoord.x & 3) == 0 && (pixelCoord.y & 3) == 0) {
            float density = 0.0;
            float separation = 0.0;
            float invSigmaK2 = 1.0 / u_SigmaK2;
            float cutoff2 = (u_MuK + 3.0 * sqrt(u_SigmaK2));
            cutoff2 *= cutoff2;
            
            // Sample only a subset of particles for field viz
            int step = max(1, u_NumParticles / 100);  // Max 100 samples
            for (int i = 0; i < u_NumParticles; i += step) {
                float mass = READ_PARTICLE_MASS(i);
                if (mass < 0.01) continue;
                
                vec2 ppos = READ_PARTICLE_POS(i);
                float d2 = wrappedDist2(worldPos, ppos);
                
                if (d2 < cutoff2) {
                    float dist = sqrt(d2);
                    float r_diff = dist - u_MuK;
                    density += u_Wk * exp(-r_diff * r_diff * invSigmaK2) * mass * float(step);
                }
                if (d2 < 1.0) {
                    float dist = sqrt(d2);
                    float prox = 1.0 - dist;
                    separation += 0.5 * prox * prox * float(step);
                }
            }
            
            float fieldVal = 0.0;
            vec3 fieldColor = vec3(0.0, 0.3, 0.8);
            
            if (u_FieldType == 1) {
                fieldVal = min(density * 2.0, 1.0);
                fieldColor = vec3(0.0, 0.3, 0.8);
            } else if (u_FieldType == 2) {
                fieldVal = min(separation, 1.0);
                fieldColor = vec3(0.8, 0.2, 0.0);
            } else if (u_FieldType == 3) {
                float u_diff = density - u_MuG;
                float growth = exp(-u_diff * u_diff / u_SigmaG2);
                fieldVal = growth;
                fieldColor = vec3(0.0, 0.8, 0.3);
            } else if (u_FieldType == 4) {
                float u_diff = density - u_MuG;
                float growth = exp(-u_diff * u_diff / u_SigmaG2);
                float e = separation - growth;
                if (e > 0.0) {
                    fieldVal = min(e, 1.0);
                    fieldColor = vec3(0.8, 0.0, 0.0);
                } else {
                    fieldVal = min(-e, 1.0);
                    fieldColor = vec3(0.0, 0.8, 0.3);
                }
            }
            
            color = mix(BACKGROUND, fieldColor, fieldVal * 0.4);
        } else {
            // For non-sampled pixels, just use background with slight tint
            color = BACKGROUND * 1.1;
        }
    }
    
    // === FOOD RENDERING ===
    // Show food as glowing green spots
    if (u_ShowFood) {
        // Convert world position to food texture UV
        // World goes from -worldWidth/2 to +worldWidth/2
        vec2 foodUV = (worldPos + vec2(u_WorldWidth, u_WorldHeight) * 0.5) / vec2(u_WorldWidth, u_WorldHeight);
        
        // Wrap for toroidal display
        foodUV = fract(foodUV);
        
        // Sample food texture
        vec4 foodData = texture(u_FoodTexture, foodUV);
        float foodAmount = foodData.r;
        float freshness = foodData.g;
        
        if (foodAmount > 0.01) {
            // Food color: bright green/yellow based on freshness
            vec3 foodColor = mix(vec3(0.3, 0.8, 0.1), vec3(0.9, 0.9, 0.2), freshness);
            
            // Add glow effect based on food density
            float foodGlow = foodAmount * 0.6;
            
            // Sparkle effect for fresh food
            if (freshness > 0.5) {
                vec2 sparkleUV = foodUV * float(u_FoodGridSize);
                float sparkle = sin(sparkleUV.x * 3.14159 * 2.0) * sin(sparkleUV.y * 3.14159 * 2.0);
                sparkle = max(0.0, sparkle) * freshness * 0.3;
                foodGlow += sparkle;
            }
            
            color = mix(color, foodColor, foodGlow);
        }
    }
    
    // === OPTIMIZED PARTICLE RENDERING ===
    float minDist2 = 1000.0;
    vec3 closestColor = vec3(1.0);
    float closestEnergy = 0.0;
    
    float glowRadius = 0.5 / u_Zoom;
    float glowRadius2 = glowRadius * glowRadius;
    
    // Only check particles within potential glow range
    for (int i = 0; i < u_NumParticles; i++) {
        float mass = READ_PARTICLE_MASS(i);
        if (mass < 0.01) continue;
        
        vec2 ppos = READ_PARTICLE_POS(i);
        float d2 = wrappedDist2(worldPos, ppos);
        
        // Early exit if beyond glow range and we already found something
        if (d2 > glowRadius2 && minDist2 < glowRadius2) continue;
        
        if (d2 < minDist2) {
            minDist2 = d2;
            closestEnergy = mass;
            closestColor = channelColor(READ_PARTICLE_SPECIES(i), mass);
        }
    }
    
    float minDist = sqrt(minDist2);
    float particleRadius = 0.15 / u_Zoom;
    
    // Glow
    if (minDist < glowRadius) {
        float glow = 1.0 - minDist / glowRadius;
        glow *= glow;
        color = mix(color, closestColor * 0.5, glow * closestEnergy * 0.5);
    }
    
    // Core
    if (minDist < particleRadius) {
        float core = 1.0 - minDist / particleRadius;
        core = sqrt(core);
        color = mix(color, closestColor, core);
    }
    
    // Bright center
    if (minDist < particleRadius * 0.3) {
        color = mix(color, vec3(1.0), 0.8);
    }
    
    FragColor = vec4(color, 1.0);
}
