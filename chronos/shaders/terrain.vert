#version 450 core

/*
 * 3D Terrain Vertex Shader
 * 
 * Generates a terrain mesh and displaces vertices based on heightmap.
 * Uses instancing for efficient grid generation.
 */

layout(location = 0) in vec2 aGridPos;  // Grid position (0-1 range)

out vec3 vWorldPos;
out vec3 vNormal;
out vec2 vUV;
out float vHeight;
out vec4 vFieldData;  // height, density, species, energy

uniform sampler2D u_Heightmap;

uniform mat4 u_ViewProjection;
uniform float u_WorldWidth;
uniform float u_WorldHeight;
uniform float u_MaxHeight;
uniform float u_TranslateX;
uniform float u_TranslateY;
uniform float u_Zoom;

// Sample height with filtering
vec4 sampleHeightmap(vec2 uv) {
    return texture(u_Heightmap, uv);
}

void main() {
    vUV = aGridPos;
    
    // Map grid to world coordinates
    vec2 worldXZ = vec2(
        (aGridPos.x - 0.5) * u_WorldWidth,
        (aGridPos.y - 0.5) * u_WorldHeight
    );
    
    // Apply view transform
    worldXZ = worldXZ / u_Zoom - vec2(u_TranslateX, u_TranslateY);
    
    // Sample heightmap
    vFieldData = sampleHeightmap(aGridPos);
    vHeight = vFieldData.r;
    
    // Calculate normal using central differences
    float texelX = 1.0 / textureSize(u_Heightmap, 0).x;
    float texelY = 1.0 / textureSize(u_Heightmap, 0).y;
    
    float hL = sampleHeightmap(aGridPos + vec2(-texelX, 0)).r;
    float hR = sampleHeightmap(aGridPos + vec2( texelX, 0)).r;
    float hD = sampleHeightmap(aGridPos + vec2(0, -texelY)).r;
    float hU = sampleHeightmap(aGridPos + vec2(0,  texelY)).r;
    
    vec3 normal = normalize(vec3(
        (hL - hR) * u_MaxHeight,
        2.0,
        (hD - hU) * u_MaxHeight
    ));
    vNormal = normal;
    
    // Create 3D position with height displacement
    vec3 pos = vec3(worldXZ.x, vHeight * u_MaxHeight, worldXZ.y);
    vWorldPos = pos;
    
    gl_Position = u_ViewProjection * vec4(pos, 1.0);
}
