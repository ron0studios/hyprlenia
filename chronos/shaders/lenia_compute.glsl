#version 430 core

layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba32f, binding = 0) uniform image2D imgInput;
layout(rgba32f, binding = 1) uniform image2D imgOutput;

uniform float R;      // Kernel radius
uniform float dt;     // Time step = 1/T

// Species parameters (species10 - tri-color ghosts style)
const int NUM_KERNELS = 3;

// Kernel parameters
const float betaLen[3] = float[3](2.0, 3.0, 1.0);
const float beta0[3] = float[3](0.25, 1.0, 1.0);
const float beta1[3] = float[3](1.0, 0.75, 0.0);
const float beta2[3] = float[3](0.0, 0.75, 0.0);

// Growth function parameters
const float mu[3] = float[3](0.16, 0.22, 0.28);
const float sigma[3] = float[3](0.105, 0.042, 0.025);
const float eta[3] = float[3](2.0, 2.0, 2.0);
const float relR[3] = float[3](1.0, 1.0, 1.0);

// Source and destination channels
const int srcCh[3] = int[3](0, 0, 0);  // All kernels read from channel 0
const int dstCh[3] = int[3](0, 0, 0);  // All kernels write to channel 0

// Kernel ring center and width
const float kmu = 0.5;
const float ksigma = 0.15;

// Bell-shaped Gaussian curve
float bell(float x, float m, float s) {
    float diff = x - m;
    return exp(-diff * diff / (2.0 * s * s));
}

// Get kernel weight for given normalized radius r (0 to 1)
float getWeight(float r, int kernelIdx) {
    float Br = betaLen[kernelIdx] * r / relR[kernelIdx];
    int BrInt = int(Br);
    
    float height = 0.0;
    if (BrInt == 0) height = beta0[kernelIdx];
    else if (BrInt == 1) height = beta1[kernelIdx];
    else if (BrInt == 2) height = beta2[kernelIdx];
    
    float modBr = fract(Br);
    return height * bell(modBr, kmu, ksigma);
}

// Get value from source channel
float getSrc(vec4 val, int ch) {
    if (ch == 0) return val.r;
    if (ch == 1) return val.g;
    return val.b;
}

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(imgInput);
    
    if (pos.x >= size.x || pos.y >= size.y) return;
    
    int intR = int(ceil(R));
    
    // Accumulate weighted sums for each kernel
    float sum[3] = float[3](0.0, 0.0, 0.0);
    float total[3] = float[3](0.0001, 0.0001, 0.0001);
    
    // Sample center
    vec4 centerVal = imageLoad(imgInput, pos);
    for (int k = 0; k < NUM_KERNELS; k++) {
        float w = getWeight(0.0, k);
        sum[k] += getSrc(centerVal, srcCh[k]) * w;
        total[k] += w;
    }
    
    // Orthogonal directions
    for (int x = 1; x <= intR; x++) {
        float r = float(x) / R;
        if (r > 1.0) break;
        
        ivec2 offsets[4] = ivec2[4](
            ivec2(x, 0), ivec2(-x, 0), ivec2(0, x), ivec2(0, -x)
        );
        
        for (int k = 0; k < NUM_KERNELS; k++) {
            float w = getWeight(r, k);
            for (int i = 0; i < 4; i++) {
                ivec2 np = (pos + offsets[i] + size) % size;
                vec4 val = imageLoad(imgInput, np);
                sum[k] += getSrc(val, srcCh[k]) * w;
                total[k] += w;
            }
        }
    }
    
    // Diagonal directions
    for (int x = 1; x <= intR; x++) {
        float r = sqrt(2.0) * float(x) / R;
        if (r > 1.0) break;
        
        ivec2 offsets[4] = ivec2[4](
            ivec2(x, x), ivec2(x, -x), ivec2(-x, x), ivec2(-x, -x)
        );
        
        for (int k = 0; k < NUM_KERNELS; k++) {
            float w = getWeight(r, k);
            for (int i = 0; i < 4; i++) {
                ivec2 np = (pos + offsets[i] + size) % size;
                vec4 val = imageLoad(imgInput, np);
                sum[k] += getSrc(val, srcCh[k]) * w;
                total[k] += w;
            }
        }
    }
    
    // Other octants
    for (int y = 1; y <= intR - 1; y++) {
        for (int x = y + 1; x <= intR; x++) {
            float r = sqrt(float(x*x + y*y)) / R;
            if (r > 1.0) continue;
            
            ivec2 offsets[8] = ivec2[8](
                ivec2(x, y), ivec2(x, -y), ivec2(-x, y), ivec2(-x, -y),
                ivec2(y, x), ivec2(y, -x), ivec2(-y, x), ivec2(-y, -x)
            );
            
            for (int k = 0; k < NUM_KERNELS; k++) {
                float w = getWeight(r, k);
                for (int i = 0; i < 8; i++) {
                    ivec2 np = (pos + offsets[i] + size) % size;
                    vec4 val = imageLoad(imgInput, np);
                    sum[k] += getSrc(val, srcCh[k]) * w;
                    total[k] += w;
                }
            }
        }
    }
    
    // Calculate averages and growth
    float growthR = 0.0, growthG = 0.0, growthB = 0.0;
    
    for (int k = 0; k < NUM_KERNELS; k++) {
        float avg = sum[k] / total[k];
        float g = eta[k] * (2.0 * bell(avg, mu[k], sigma[k]) - 1.0);
        
        // Add growth to destination channel
        if (dstCh[k] == 0) growthR += g;
        else if (dstCh[k] == 1) growthG += g;
        else growthB += g;
    }
    
    // Update values
    vec4 current = imageLoad(imgInput, pos);
    vec4 newVal = current + dt * vec4(growthR, growthG, growthB, 0.0);
    newVal = clamp(newVal, 0.0, 1.0);
    
    imageStore(imgOutput, pos, newVal);
}
