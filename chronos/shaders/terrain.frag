#version 450 core

/*
 * 3D Terrain Fragment Shader
 * 
 * Renders the terrain with smooth lighting, glow effects,
 * and color based on particle density and species.
 */

in vec3 vWorldPos;
in vec3 vNormal;
in vec2 vUV;
in float vHeight;
in vec4 vFieldData;  // height, density, species, energy

out vec4 FragColor;

uniform vec3 u_CameraPos;
uniform vec3 u_LightDir;
uniform float u_Time;
uniform float u_GlowIntensity;
uniform float u_AmbientStrength;
uniform bool u_ShowWireframe;

// Color palette
const vec3 BASE_COLOR = vec3(0.05, 0.08, 0.15);     // Visible dark blue
const vec3 GRID_COLOR = vec3(0.1, 0.2, 0.3);        // Grid lines
const vec3 GLOW_COLOR_1 = vec3(0.1, 0.5, 1.0);      // Blue glow
const vec3 GLOW_COLOR_2 = vec3(0.0, 1.0, 0.5);      // Cyan-green glow
const vec3 GLOW_COLOR_3 = vec3(1.0, 0.3, 0.5);      // Pink glow
const vec3 HIGHLIGHT = vec3(1.0, 1.0, 1.0);         // White highlights

// HSL to RGB for species coloring
vec3 hsl2rgb(vec3 c) {
    vec3 rgb = clamp(abs(mod(c.x * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
    return c.z + c.y * (rgb - 0.5) * (1.0 - abs(2.0 * c.z - 1.0));
}

vec3 speciesColor(float species) {
    float hue = mod(species * 0.3, 1.0);
    return hsl2rgb(vec3(hue, 0.8, 0.6));
}

void main() {
    vec3 normal = normalize(vNormal);
    vec3 viewDir = normalize(u_CameraPos - vWorldPos);
    vec3 lightDir = normalize(u_LightDir);
    
    // Extract field data
    float height = vFieldData.r;
    float density = vFieldData.g;
    float species = vFieldData.b * 3.0;  // Reconstruct species
    float energy = vFieldData.a;
    
    // === BASE LIGHTING ===
    float NdotL = max(dot(normal, lightDir), 0.0);
    float ambient = u_AmbientStrength;
    float diffuse = NdotL * 0.6;
    
    // Rim lighting for edge glow
    float rim = 1.0 - max(dot(normal, viewDir), 0.0);
    rim = pow(rim, 3.0);
    
    // === COLOR CALCULATION ===
    vec3 color = BASE_COLOR;
    
    // Add height-based color
    float heightFactor = smoothstep(0.0, 2.0, height);
    vec3 heightColor = mix(GLOW_COLOR_1, GLOW_COLOR_2, heightFactor);
    
    // Species-based color influence
    vec3 specColor = speciesColor(species);
    
    // Blend based on density and energy
    float colorInfluence = smoothstep(0.0, 0.5, density) * energy;
    color = mix(color, specColor * 0.5 + heightColor * 0.5, colorInfluence);
    
    // === GLOW EFFECT ===
    float glowStrength = height * u_GlowIntensity;
    vec3 glowColor = mix(heightColor, specColor, 0.5);
    color += glowColor * glowStrength * 0.5;
    
    // Rim glow
    color += glowColor * rim * glowStrength * 0.3;
    
    // === LIGHTING APPLICATION ===
    vec3 finalColor = color * (ambient + diffuse);
    
    // Specular highlights on peaks
    vec3 halfDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfDir), 0.0), 32.0);
    finalColor += HIGHLIGHT * spec * heightFactor * 0.5;
    
    // === SUBTLE ANIMATION ===
    float pulse = sin(u_Time * 2.0 + vWorldPos.x * 0.5 + vWorldPos.z * 0.5) * 0.5 + 0.5;
    finalColor += glowColor * pulse * glowStrength * 0.1;
    
    // === ALWAYS VISIBLE GROUND GRID ===
    // Major grid lines (8x8 divisions)
    vec2 majorGrid = abs(fract(vUV * 8.0 - 0.5) - 0.5);
    float majorLine = 1.0 - smoothstep(0.0, 0.03, min(majorGrid.x, majorGrid.y));
    finalColor += GRID_COLOR * majorLine * 0.4;
    
    // Minor grid lines (32x32 divisions)
    vec2 minorGrid = abs(fract(vUV * 32.0 - 0.5) - 0.5);
    float minorLine = 1.0 - smoothstep(0.0, 0.04, min(minorGrid.x, minorGrid.y));
    finalColor += GRID_COLOR * minorLine * 0.15;
    
    // Edge glow on terrain boundary
    vec2 edgeDist = min(vUV, 1.0 - vUV);
    float edgeGlow = 1.0 - smoothstep(0.0, 0.05, min(edgeDist.x, edgeDist.y));
    finalColor += vec3(0.2, 0.5, 1.0) * edgeGlow * 0.5;
    
    // === WIREFRAME (optional, enhanced) ===
    if (u_ShowWireframe) {
        vec2 grid = fract(vUV * 64.0);
        float wireframe = 1.0 - smoothstep(0.0, 0.05, min(grid.x, grid.y));
        wireframe *= 1.0 - smoothstep(0.95, 1.0, max(grid.x, grid.y));
        finalColor = mix(finalColor, vec3(0.3, 0.6, 1.0), wireframe * 0.5);
    }
    
    // Fog for depth (reduced intensity)
    float fogDist = length(vWorldPos - u_CameraPos);
    float fog = 1.0 - exp(-fogDist * 0.015);
    vec3 fogColor = vec3(0.02, 0.04, 0.08);
    finalColor = mix(finalColor, fogColor, fog * 0.4);
    
    // Ensure minimum visibility
    finalColor = max(finalColor, BASE_COLOR * 0.3);
    
    FragColor = vec4(finalColor, 1.0);
}
