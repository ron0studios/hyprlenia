#version 460 core

/*
 * 3D Particle Vertex Shader
 *
 * Renders particles as point sprites positioned in true 3D space.
 * Particle layout (14 floats):
 *   0-2: position (x, y, z)
 *   3-5: velocity (vx, vy, vz)
 *   6: energy
 *   7: species
 *   8: age
 *   9-13: dna[5]
 */

layout(std430, binding = 0) readonly buffer Particles {
    float particles[];
};

uniform mat4 u_ViewProjection;
uniform int u_NumParticles;
uniform float u_WorldWidth;
uniform float u_WorldHeight;
uniform float u_WorldDepth;
uniform float u_ParticleSize;
uniform float u_TranslateX;
uniform float u_TranslateY;
uniform float u_TranslateZ;
uniform float u_Zoom;
uniform vec3 u_CameraPos;

out float vEnergy;
out float vSpecies;
out vec3 vWorldPos;

// Particle data accessors (14 floats per particle)
#define READ_PARTICLE_POS(i) vec3(particles[(i) * 14], particles[(i) * 14 + 1], particles[(i) * 14 + 2])
#define READ_PARTICLE_ENERGY(i) particles[(i) * 14 + 6]
#define READ_PARTICLE_SPECIES(i) particles[(i) * 14 + 7]

void main() {
    int idx = gl_VertexID;

    // Read 3D position directly
    vec3 pos = READ_PARTICLE_POS(idx);
    vEnergy = READ_PARTICLE_ENERGY(idx);
    vSpecies = READ_PARTICLE_SPECIES(idx);

    // Skip dead particles
    if (vEnergy < 0.01) {
        gl_Position = vec4(-1000.0, -1000.0, -1000.0, 1.0);
        gl_PointSize = 0.0;
        return;
    }

    // Apply view transform (zoom and translation)
    vec3 viewPos = pos / u_Zoom - vec3(u_TranslateX, u_TranslateY, u_TranslateZ);

    // World position for fragment shader (Y is up in OpenGL)
    // Map: simulation (x,y,z) -> rendering (x, z, -y) so Y becomes up
    vec3 worldPos = vec3(viewPos.x, viewPos.z, -viewPos.y);
    vWorldPos = worldPos;

    gl_Position = u_ViewProjection * vec4(worldPos, 1.0);

    // Point size based on distance from camera and energy
    float dist = length(u_CameraPos - worldPos);
    gl_PointSize = u_ParticleSize * vEnergy / (dist * 0.05 + 1.0);
}
