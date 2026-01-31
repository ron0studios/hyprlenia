#version 430 core

out vec4 FragColor;
in vec2 TexCoord;

uniform sampler2D leniaTexture;
uniform sampler2D bloomTexture;
uniform float time;
uniform float bloomIntensity;
uniform float glowPower;

// Color mapping from Shadertoy reference
vec3 GreyToColorMix(float val) {
    vec3 retVal;
    
    // Red channel
    if (val < 0.5) {
        retVal.r = 0.02;
    } else if (val < 0.7529) {
        retVal.r = (1.0 / 0.25) * (val - 0.5);
    } else {
        retVal.r = 1.0;
    }
    
    // Green channel
    if (val < 0.25) {
        retVal.g = (1.0 / 0.30) * val;
    } else if (val < 0.752) {
        retVal.g = 1.0;
    } else {
        retVal.g = -(1.0 / 0.25) * (val - 0.752) + 1.0;
    }
    
    // Blue channel
    if (val < 0.25) {
        retVal.b = 0.1;
    } else if (val < 0.5) {
        retVal.b = -(1.0 / 0.25) * (val - 0.752) + 1.0;
    } else {
        retVal.b = 0.0;
    }
    
    return retVal;
}

// Sample with distance weighting for glow effect
float sampleAround(int dist, vec2 uv, vec2 texelSize) {
    float val = 0.0;
    for (int u = -dist; u <= dist; ++u) {
        for (int v = -dist; v <= dist; ++v) {
            float weight = 1.0 / max(float(u*u + v*v), 1.0);
            vec2 offset = vec2(float(u), float(v)) * texelSize;
            val += clamp(texture(leniaTexture, uv + offset).r, 0.0, 1.0) * weight;
        }
    }
    return val;
}

void main() {
    vec2 texSize = vec2(textureSize(leniaTexture, 0));
    vec2 texelSize = 1.0 / texSize;
    
    vec4 val = texture(leniaTexture, TexCoord);
    vec4 bloom = texture(bloomTexture, TexCoord);
    
    // Sample with glow effect (distance-weighted neighborhood)
    float glowSample = sampleAround(1, TexCoord, texelSize) / 10.0;
    
    // Get colorized version
    vec3 colorized = GreyToColorMix(glowSample);
    
    // Combine with raw cell value for contrast
    float cellVal = val.r;
    vec3 color = colorized + vec3(cellVal * 3.0, 0.0, 0.0) * 0.3;
    
    // Apply color grading like reference
    color = color * vec3(0.7, 0.4, 0.45) * 3.0;
    
    // Add subtle bloom
    color += bloom.rgb * bloomIntensity * 0.3;
    
    // Subtle vignette
    vec2 uv = TexCoord * 2.0 - 1.0;
    float vignette = 1.0 - dot(uv, uv) * 0.15;
    color *= vignette;
    
    // Tone mapping
    color = color / (color + vec3(1.0));
    
    // Gamma correction
    color = pow(color, vec3(1.0 / 2.2));
    
    FragColor = vec4(color, 1.0);
}
