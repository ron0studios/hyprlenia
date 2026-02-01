#version 460 core

 

in float vEnergy;
in float vSpecies;
in float vPotential;
in vec3 vWorldPos;

out vec4 FragColor;

uniform vec3 u_CameraPos;
uniform float u_Time;
uniform float u_MuG;
uniform float u_SigmaG2;

vec3 hsl2rgb(vec3 c) {
    vec3 rgb = clamp(abs(mod(c.x * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
    return c.z + c.y * (rgb - 0.5) * (1.0 - abs(2.0 * c.z - 1.0));
}

// Generate color based on Potential (U) and Growth (G)
// Similar to Lenia color maps
vec3 getBlobColor(float potential, float energy) {
    float u_diff = potential - u_MuG;
    float growth = exp(-u_diff * u_diff / u_SigmaG2); // G(U)
    
    // Map G to a color gradient
    // Low G (Decay) -> Blue/Purple
    // High G (Growth) -> Green/Cyan
    // Very High Energy -> White/Yellow
    
    vec3 colDecay = vec3(0.1, 0.0, 0.3); // Dark Purple
    vec3 colStable = vec3(0.0, 0.4, 0.8); // Blue
    vec3 colGrowth = vec3(0.2, 0.9, 0.5); // Neon Green
    
    vec3 color;
    if (growth < 0.5) {
        color = mix(colDecay, colStable, growth * 2.0);
    } else {
        color = mix(colStable, colGrowth, (growth - 0.5) * 2.0);
    }
    
    // Energy boosts brightness/saturation
    color += vec3(0.8, 0.8, 0.6) * energy * 0.5;
    
    return color;
}

void main() {
    
    if (vEnergy < 0.01) {
        discard;
    }
    
    
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float dist = length(coord);
    
    if (dist > 1.0) {
        discard;
    }
    
    
    float sphere = 1.0 - dist;
    sphere = pow(sphere, 1.5);
    
    
    // vec3 color = speciesColor(vSpecies, vEnergy);
    vec3 color = getBlobColor(vPotential, vEnergy);
    
    
    float glow = exp(-dist * 2.0) * vEnergy;
    color += vec3(0.5, 0.8, 1.0) * glow * 0.3;
    
    
    float pulse = sin(u_Time * 2.0 + vPotential * 10.0) * 0.05 + 0.95;
    color *= pulse;
    
    
    float alpha = sphere * vEnergy;
    
    FragColor = vec4(color, alpha);
}
