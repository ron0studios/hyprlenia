#version 460 core

/*
 * 3D Particle Vertex Shader
 * 
 * Renders particles as point sprites positioned in 3D space.
 * Height is sampled from the heightmap.
 */

layout(std430, binding = 0) readonly buffer Particles {
    float particles[];
};

uniform mat4 u_ViewProjection;
uniform sampler2D u_Heightmap;
uniform float u_WorldWidth;
uniform float u_WorldHeight;
uniform float u_MaxHeight;
uniform float u_ParticleSize;
uniform float u_TranslateX;
uniform float u_TranslateY;
uniform float u_Zoom;

out float vEnergy;
out float vSpecies;
out vec3 vWorldPos;

#define READ_PARTICLE_POS(i) vec2(particles[(i) * 12], particles[(i) * 12 + 1])
#define READ_PARTICLE_ENERGY(i) particles[(i) * 12 + 4]
#define READ_PARTICLE_SPECIES(i) particles[(i) * 12 + 5]

void main() {
    int idx = gl_VertexID;
    
    vec2 pos = READ_PARTICLE_POS(idx);
    vEnergy = READ_PARTICLE_ENERGY(idx);
    vSpecies = READ_PARTICLE_SPECIES(idx);
    
    // Skip dead particles
    if (vEnergy < 0.01) {
        gl_Position = vec4(-1000.0, -1000.0, -1000.0, 1.0);
        gl_PointSize = 0.0;
        return;
    }
    
    // Map particle position to UV for heightmap sampling
    vec2 uv = vec2(
        pos.x / u_WorldWidth + 0.5,
        pos.y / u_WorldHeight + 0.5
    );
    
    // Sample height
    float height = texture(u_Heightmap, uv).r;
    
    // Apply view transform
    vec2 viewPos = pos / u_Zoom - vec2(u_TranslateX, u_TranslateY);
    
    // Create 3D position
    vec3 worldPos = vec3(viewPos.x, height * u_MaxHeight + 0.3, viewPos.y);
    vWorldPos = worldPos;
    
    gl_Position = u_ViewProjection * vec4(worldPos, 1.0);
    
    // Point size based on distance and energy
    float dist = length(gl_Position.xyz);
    gl_PointSize = u_ParticleSize * vEnergy / (dist * 0.1 + 1.0);
}
