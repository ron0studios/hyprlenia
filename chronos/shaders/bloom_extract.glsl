#version 430 core

layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba32f, binding = 0) uniform image2D imgInput;
layout(rgba32f, binding = 1) uniform image2D imgOutput;

uniform float threshold;

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(imgInput);
    
    if (pos.x >= size.x || pos.y >= size.y) return;
    
    vec4 color = imageLoad(imgInput, pos);
    
    // Extract bright areas for bloom
    float brightness = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
    
    vec4 result = vec4(0.0);
    if (brightness > threshold) {
        result = color * (brightness - threshold) / (1.0 - threshold);
    }
    
    imageStore(imgOutput, pos, result);
}
