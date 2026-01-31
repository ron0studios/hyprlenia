#version 430 core

layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba32f, binding = 0) uniform image2D imgInput;
layout(rgba32f, binding = 1) uniform image2D imgOutput;

uniform vec2 texelSize;
uniform bool horizontal;

// 9-tap Gaussian weights for smooth bloom
const float weights[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(imgInput);
    
    if (pos.x >= size.x || pos.y >= size.y) return;
    
    vec4 result = imageLoad(imgInput, pos) * weights[0];
    
    ivec2 dir = horizontal ? ivec2(1, 0) : ivec2(0, 1);
    
    for (int i = 1; i < 5; i++) {
        ivec2 offset = dir * i;
        ivec2 p1 = clamp(pos + offset, ivec2(0), size - 1);
        ivec2 p2 = clamp(pos - offset, ivec2(0), size - 1);
        result += imageLoad(imgInput, p1) * weights[i];
        result += imageLoad(imgInput, p2) * weights[i];
    }
    
    imageStore(imgOutput, pos, result);
}
