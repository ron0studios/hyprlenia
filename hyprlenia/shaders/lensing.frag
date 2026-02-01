#version 460 core

in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D u_SceneTexture;
uniform vec2 u_BlackHoleScreenPos;  // Black hole position in screen space (0-1)
uniform float u_SchwarzschildRadius; // Apparent size of event horizon in screen units
uniform float u_LensingStrength;     // Overall lensing intensity multiplier
uniform float u_AspectRatio;
uniform bool u_ShowAccretionDisk;
uniform float u_DiskInnerRadius;     // Inner radius of accretion disk (multiples of r_s)
uniform float u_DiskOuterRadius;     // Outer radius of accretion disk
uniform float u_Time;

const float PI = 3.14159265359;

// HSL to RGB for accretion disk coloring
vec3 hsl2rgb(vec3 c) {
    vec3 rgb = clamp(abs(mod(c.x * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
    return c.z + c.y * (rgb - 0.5) * (1.0 - abs(2.0 * c.z - 1.0));
}

// Compute gravitational lensing deflection
// Based on Schwarzschild geometry: deflection angle ~ 2*r_s / b
// where b is the impact parameter (distance from black hole center)
vec2 computeLensedUV(vec2 uv, vec2 bhPos, float r_s, float strength) {
    // Correct for aspect ratio
    vec2 correctedUV = uv;
    correctedUV.x *= u_AspectRatio;
    vec2 correctedBH = bhPos;
    correctedBH.x *= u_AspectRatio;

    vec2 delta = correctedUV - correctedBH;
    float dist = length(delta);

    // Avoid division by zero and singularity at center
    if (dist < 0.001) {
        return vec2(-1.0); // Will be rendered as black hole
    }

    vec2 dir = delta / dist;

    // Schwarzschild deflection: light bends toward the black hole
    // Deflection angle approximation: alpha = 2 * r_s / b
    // For stronger effect near the photon sphere (r = 1.5 * r_s), we enhance it

    float impactParam = dist;
    float photonSphereRadius = 1.5 * r_s;

    // Enhanced lensing formula that creates Einstein ring effect
    float deflection = 0.0;

    if (impactParam > r_s) {
        // Standard weak-field deflection with enhancement near photon sphere
        float baseDeflection = (2.0 * r_s) / impactParam;

        // Enhance near photon sphere
        float photonFactor = 1.0 + 2.0 * exp(-pow((impactParam - photonSphereRadius) / (0.5 * r_s), 2.0));

        deflection = baseDeflection * photonFactor * strength;
    }

    // Apply deflection - light bends TOWARD the black hole
    // So we sample from a position further from the center
    vec2 lensedDelta = delta + dir * deflection;

    // Convert back to screen coordinates
    vec2 lensedUV;
    lensedUV.x = correctedBH.x + lensedDelta.x;
    lensedUV.y = correctedBH.y + lensedDelta.y;
    lensedUV.x /= u_AspectRatio;

    return lensedUV;
}

// Accretion disk color based on radius and angle
vec4 accretionDiskColor(float r, float angle, float innerR, float outerR) {
    if (r < innerR || r > outerR) return vec4(0.0);

    float t = (r - innerR) / (outerR - innerR);

    // Temperature decreases with radius (inner disk is hotter)
    // Hot inner: white/blue, cooler outer: orange/red
    float temperature = 1.0 - t;

    // Doppler effect simulation (one side brighter due to rotation)
    float doppler = 0.5 + 0.5 * sin(angle + u_Time * 0.5);

    // Color gradient from hot (white/yellow) to cool (red/orange)
    vec3 hotColor = vec3(1.0, 0.9, 0.7);
    vec3 coolColor = vec3(1.0, 0.3, 0.1);
    vec3 diskColor = mix(coolColor, hotColor, temperature);

    // Apply doppler brightening
    diskColor *= 0.7 + 0.6 * doppler;

    // Intensity falls off with radius
    float intensity = pow(1.0 - t, 0.5) * (0.8 + 0.2 * doppler);

    // Add some turbulence/structure
    float turbulence = sin(angle * 8.0 + r * 20.0 + u_Time) * 0.1 + 0.9;
    intensity *= turbulence;

    return vec4(diskColor * intensity, intensity * 0.9);
}

void main() {
    vec2 uv = TexCoord;

    // Calculate distance to black hole center (aspect-corrected)
    vec2 correctedUV = uv;
    correctedUV.x *= u_AspectRatio;
    vec2 correctedBH = u_BlackHoleScreenPos;
    correctedBH.x *= u_AspectRatio;

    vec2 toBH = correctedUV - correctedBH;
    float distToBH = length(toBH);
    float angle = atan(toBH.y, toBH.x);

    float r_s = u_SchwarzschildRadius;

    // Inside event horizon - pure black
    if (distToBH < r_s) {
        FragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // Compute lensed UV coordinates
    vec2 lensedUV = computeLensedUV(uv, u_BlackHoleScreenPos, r_s, u_LensingStrength);

    // Sample scene at lensed coordinates
    vec4 sceneColor = vec4(0.0);
    if (lensedUV.x >= 0.0 && lensedUV.x <= 1.0 && lensedUV.y >= 0.0 && lensedUV.y <= 1.0) {
        sceneColor = texture(u_SceneTexture, lensedUV);
    }

    // Add accretion disk if enabled
    vec4 diskColor = vec4(0.0);
    if (u_ShowAccretionDisk) {
        float innerR = r_s * u_DiskInnerRadius;
        float outerR = r_s * u_DiskOuterRadius;
        diskColor = accretionDiskColor(distToBH, angle, innerR, outerR);
    }

    // Photon sphere glow (light orbiting at r = 1.5 * r_s)
    float photonSphereR = 1.5 * r_s;
    float photonGlow = exp(-pow((distToBH - photonSphereR) / (0.15 * r_s), 2.0)) * 0.3;
    vec3 photonColor = vec3(1.0, 0.7, 0.4) * photonGlow;

    // Combine layers
    vec3 finalColor = sceneColor.rgb;

    // Add photon sphere glow
    finalColor += photonColor;

    // Blend accretion disk on top
    finalColor = mix(finalColor, diskColor.rgb, diskColor.a);

    // Slight vignette around event horizon
    float horizonFade = smoothstep(r_s, r_s * 1.2, distToBH);
    finalColor *= horizonFade;

    FragColor = vec4(finalColor, 1.0);
}
