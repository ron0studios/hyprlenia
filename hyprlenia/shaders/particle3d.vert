#version 460 core

 

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


#define READ_PARTICLE_POS(i) vec3(particles[(i) * 15], particles[(i) * 15 + 1], particles[(i) * 15 + 2])
#define READ_PARTICLE_ENERGY(i) particles[(i) * 15 + 6]
#define READ_PARTICLE_SPECIES(i) particles[(i) * 15 + 7]

void main() {
    int idx = gl_VertexID;

    
    vec3 pos = READ_PARTICLE_POS(idx);
    vEnergy = READ_PARTICLE_ENERGY(idx);
    vSpecies = READ_PARTICLE_SPECIES(idx);

    
    if (vEnergy < 0.01) {
        gl_Position = vec4(-1000.0, -1000.0, -1000.0, 1.0);
        gl_PointSize = 0.0;
        return;
    }

    
    vec3 viewPos = pos / u_Zoom - vec3(u_TranslateX, u_TranslateY, u_TranslateZ);

    
    
    vec3 worldPos = vec3(viewPos.x, viewPos.z, -viewPos.y);
    vWorldPos = worldPos;

    gl_Position = u_ViewProjection * vec4(worldPos, 1.0);

    
    float dist = length(u_CameraPos - worldPos);
    gl_PointSize = u_ParticleSize * vEnergy / (dist * 0.05 + 1.0);
}
