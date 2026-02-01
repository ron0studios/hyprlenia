#version 450 core

 

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


uniform sampler2D u_FoodTexture;
uniform bool u_ShowFood;
uniform int u_FoodGridSize;

const vec3 BACKGROUND = vec3(0.005, 0.02, 0.05);


vec3 hsl2rgb(vec3 c) {
    vec3 rgb = clamp(abs(mod(c.x * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
    return c.z + c.y * (rgb - 0.5) * (1.0 - abs(2.0 * c.z - 1.0));
}

vec3 speciesColor(float species, float energy) {
    float hue = mod(species * 0.3, 1.0);
    return hsl2rgb(vec3(hue, 0.7 + energy * 0.3, 0.3 + energy * 0.4));
}


#define READ_PARTICLE_POS(i) vec2(particles[(i) * 15], particles[(i) * 15 + 1])
#define READ_PARTICLE_MASS(i) particles[(i) * 15 + 6]
#define READ_PARTICLE_SPECIES(i) particles[(i) * 15 + 7]



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
    
    
    float windowAspect = u_WindowWidth / u_WindowHeight;
    float worldAspect = u_WorldWidth / u_WorldHeight;
    
    
    vec2 scaledUV = uv;
    if (windowAspect > worldAspect) {
        
        scaledUV.x *= windowAspect / worldAspect;
    } else {
        
        scaledUV.y *= worldAspect / windowAspect;
    }
    
    
    vec2 worldPos = vec2(
        scaledUV.x * (u_WorldWidth * 0.5) / u_Zoom + u_TranslateX,
        scaledUV.y * (u_WorldHeight * 0.5) / u_Zoom + u_TranslateY
    );

    
    if (abs(worldPos.x) > u_WorldWidth * 0.5 || abs(worldPos.y) > u_WorldHeight * 0.5) {
        FragColor = vec4(BACKGROUND * 0.5, 1.0);  
        return;
    }
    
    vec3 color = BACKGROUND;
    
    
    
    if (u_ShowFields && u_FieldType > 0) {
        ivec2 pixelCoord = ivec2(gl_FragCoord.xy);
        
        
        if ((pixelCoord.x & 3) == 0 && (pixelCoord.y & 3) == 0) {
            float density = 0.0;
            float separation = 0.0;
            float invSigmaK2 = 1.0 / u_SigmaK2;
            float cutoff2 = (u_MuK + 3.0 * sqrt(u_SigmaK2));
            cutoff2 *= cutoff2;
            
            
            int step = max(1, u_NumParticles / 100);  
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
            
            color = BACKGROUND * 1.1;
        }
    }
    
    
    
    if (u_ShowFood) {
        
        
        vec2 foodUV = (worldPos + vec2(u_WorldWidth, u_WorldHeight) * 0.5) / vec2(u_WorldWidth, u_WorldHeight);
        
        
        foodUV = fract(foodUV);
        
        
        vec4 foodData = texture(u_FoodTexture, foodUV);
        float foodAmount = foodData.r;
        float freshness = foodData.g;
        
        if (foodAmount > 0.01) {
            
            vec3 foodColor = mix(vec3(0.3, 0.8, 0.1), vec3(0.9, 0.9, 0.2), freshness);
            
            
            float foodGlow = foodAmount * 0.6;
            
            
            if (freshness > 0.5) {
                vec2 sparkleUV = foodUV * float(u_FoodGridSize);
                float sparkle = sin(sparkleUV.x * 3.14159 * 2.0) * sin(sparkleUV.y * 3.14159 * 2.0);
                sparkle = max(0.0, sparkle) * freshness * 0.3;
                foodGlow += sparkle;
            }
            
            color = mix(color, foodColor, foodGlow);
        }
    }
    
    
    float minDist2 = 1000.0;
    vec3 closestColor = vec3(1.0);
    float closestEnergy = 0.0;
    
    float glowRadius = 0.5 / u_Zoom;
    float glowRadius2 = glowRadius * glowRadius;
    
    
    for (int i = 0; i < u_NumParticles; i++) {
        float mass = READ_PARTICLE_MASS(i);
        if (mass < 0.01) continue;
        
        vec2 ppos = READ_PARTICLE_POS(i);
        float d2 = wrappedDist2(worldPos, ppos);
        
        
        if (d2 > glowRadius2 && minDist2 < glowRadius2) continue;
        
        if (d2 < minDist2) {
            minDist2 = d2;
            closestEnergy = mass;
            closestColor = speciesColor(READ_PARTICLE_SPECIES(i), mass);
        }
    }
    
    float minDist = sqrt(minDist2);
    float particleRadius = 0.15 / u_Zoom;
    
    
    if (minDist < glowRadius) {
        float glow = 1.0 - minDist / glowRadius;
        glow *= glow;
        color = mix(color, closestColor * 0.5, glow * closestEnergy * 0.5);
    }
    
    
    if (minDist < particleRadius) {
        float core = 1.0 - minDist / particleRadius;
        core = sqrt(core);
        color = mix(color, closestColor, core);
    }
    
    
    if (minDist < particleRadius * 0.3) {
        color = mix(color, vec3(1.0), 0.8);
    }
    
    FragColor = vec4(color, 1.0);
}
