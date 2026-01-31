#version 430 core

layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba32f, binding = 0) uniform image2D imgState;
layout(rgba32f, binding = 1) uniform image2D imgStateOut;
layout(rg32f, binding = 2) uniform image2D imgFlow;

uniform float R;
uniform float dt;
uniform int pass;

// Rule 388 - produces self-organizing moving patterns
// From https://www.reddit.com/r/proceduralgeneration/comments/86xd9k/

// Sample cell value (binary: 0 or 1)
float cv(ivec2 pos, ivec2 size, int dx, int dy) {
    ivec2 sampCoord = (pos + ivec2(dx, dy) + size) % size;
    float o = imageLoad(imgState, sampCoord).r;
    return o > 0.5 ? 1.0 : 0.0;
}

float doCellular388(ivec2 pos, ivec2 size) {
    float outval = cv(pos, size, 0, 0);
    
    // Neighborhood 0+1: Complex ring pattern at radius ~14
    float nhd0 = 
        cv(pos,size,-14,-1)+cv(pos,size,-14,0)+cv(pos,size,-14,1)+
        cv(pos,size,-13,-4)+cv(pos,size,-13,-3)+cv(pos,size,-13,-2)+cv(pos,size,-13,2)+cv(pos,size,-13,3)+cv(pos,size,-13,4)+
        cv(pos,size,-12,-6)+cv(pos,size,-12,-5)+cv(pos,size,-12,5)+cv(pos,size,-12,6)+
        cv(pos,size,-11,-8)+cv(pos,size,-11,-7)+cv(pos,size,-11,7)+cv(pos,size,-11,8)+
        cv(pos,size,-10,-9)+cv(pos,size,-10,-1)+cv(pos,size,-10,0)+cv(pos,size,-10,1)+cv(pos,size,-10,9)+
        cv(pos,size,-9,-10)+cv(pos,size,-9,-4)+cv(pos,size,-9,-3)+cv(pos,size,-9,-2)+cv(pos,size,-9,2)+cv(pos,size,-9,3)+cv(pos,size,-9,4)+cv(pos,size,-9,10)+
        cv(pos,size,-8,-11)+cv(pos,size,-8,-6)+cv(pos,size,-8,-5)+cv(pos,size,-8,5)+cv(pos,size,-8,6)+cv(pos,size,-8,11)+
        cv(pos,size,-7,-11)+cv(pos,size,-7,-7)+cv(pos,size,-7,-2)+cv(pos,size,-7,-1)+cv(pos,size,-7,0)+cv(pos,size,-7,1)+cv(pos,size,-7,2)+cv(pos,size,-7,7)+cv(pos,size,-7,11)+
        cv(pos,size,-6,-12)+cv(pos,size,-6,-8)+cv(pos,size,-6,-4)+cv(pos,size,-6,-3)+cv(pos,size,-6,3)+cv(pos,size,-6,4)+cv(pos,size,-6,8)+cv(pos,size,-6,12)+
        cv(pos,size,-5,-12)+cv(pos,size,-5,-8)+cv(pos,size,-5,-5)+cv(pos,size,-5,-1)+cv(pos,size,-5,0)+cv(pos,size,-5,1)+cv(pos,size,-5,5)+cv(pos,size,-5,8)+cv(pos,size,-5,12)+
        cv(pos,size,-4,-13)+cv(pos,size,-4,-9)+cv(pos,size,-4,-6)+cv(pos,size,-4,-3)+cv(pos,size,-4,-2)+cv(pos,size,-4,2)+cv(pos,size,-4,3)+cv(pos,size,-4,6)+cv(pos,size,-4,9)+cv(pos,size,-4,13)+
        cv(pos,size,-3,-13)+cv(pos,size,-3,-9)+cv(pos,size,-3,-6)+cv(pos,size,-3,-4)+cv(pos,size,-3,-1)+cv(pos,size,-3,0)+cv(pos,size,-3,1)+cv(pos,size,-3,4)+cv(pos,size,-3,6)+cv(pos,size,-3,9)+cv(pos,size,-3,13)+
        cv(pos,size,-2,-13)+cv(pos,size,-2,-9)+cv(pos,size,-2,-7)+cv(pos,size,-2,-4)+cv(pos,size,-2,-2)+cv(pos,size,-2,2)+cv(pos,size,-2,4)+cv(pos,size,-2,7)+cv(pos,size,-2,9)+cv(pos,size,-2,13)+
        cv(pos,size,-1,-14)+cv(pos,size,-1,-10)+cv(pos,size,-1,-7)+cv(pos,size,-1,-5)+cv(pos,size,-1,-3)+cv(pos,size,-1,-1)+cv(pos,size,-1,0)+cv(pos,size,-1,1)+cv(pos,size,-1,3)+cv(pos,size,-1,5)+cv(pos,size,-1,7)+cv(pos,size,-1,10)+cv(pos,size,-1,14)+
        cv(pos,size,0,-14)+cv(pos,size,0,-10)+cv(pos,size,0,-7)+cv(pos,size,0,-5)+cv(pos,size,0,-3)+cv(pos,size,0,-1)+cv(pos,size,0,1)+cv(pos,size,0,3)+cv(pos,size,0,5)+cv(pos,size,0,7)+cv(pos,size,0,10)+cv(pos,size,0,14)+
        cv(pos,size,1,-14)+cv(pos,size,1,-10)+cv(pos,size,1,-7)+cv(pos,size,1,-5)+cv(pos,size,1,-3)+cv(pos,size,1,-1)+cv(pos,size,1,0)+cv(pos,size,1,1)+cv(pos,size,1,3)+cv(pos,size,1,5)+cv(pos,size,1,7)+cv(pos,size,1,10)+cv(pos,size,1,14)+
        cv(pos,size,2,-13)+cv(pos,size,2,-9)+cv(pos,size,2,-7)+cv(pos,size,2,-4)+cv(pos,size,2,-2)+cv(pos,size,2,2)+cv(pos,size,2,4)+cv(pos,size,2,7)+cv(pos,size,2,9)+cv(pos,size,2,13)+
        cv(pos,size,3,-13)+cv(pos,size,3,-9)+cv(pos,size,3,-6)+cv(pos,size,3,-4)+cv(pos,size,3,-1)+cv(pos,size,3,0)+cv(pos,size,3,1)+cv(pos,size,3,4)+cv(pos,size,3,6)+cv(pos,size,3,9)+cv(pos,size,3,13)+
        cv(pos,size,4,-13)+cv(pos,size,4,-9)+cv(pos,size,4,-6)+cv(pos,size,4,-3);
    
    float nhd1 = 
        cv(pos,size,4,-2)+cv(pos,size,4,2)+cv(pos,size,4,3)+cv(pos,size,4,6)+cv(pos,size,4,9)+cv(pos,size,4,13)+
        cv(pos,size,5,-12)+cv(pos,size,5,-8)+cv(pos,size,5,-5)+cv(pos,size,5,-1)+cv(pos,size,5,0)+cv(pos,size,5,1)+cv(pos,size,5,5)+cv(pos,size,5,8)+cv(pos,size,5,12)+
        cv(pos,size,6,-12)+cv(pos,size,6,-8)+cv(pos,size,6,-4)+cv(pos,size,6,-3)+cv(pos,size,6,3)+cv(pos,size,6,4)+cv(pos,size,6,8)+cv(pos,size,6,12)+
        cv(pos,size,7,-11)+cv(pos,size,7,-7)+cv(pos,size,7,-2)+cv(pos,size,7,-1)+cv(pos,size,7,0)+cv(pos,size,7,1)+cv(pos,size,7,2)+cv(pos,size,7,7)+cv(pos,size,7,11)+
        cv(pos,size,8,-11)+cv(pos,size,8,-6)+cv(pos,size,8,-5)+cv(pos,size,8,5)+cv(pos,size,8,6)+cv(pos,size,8,11)+
        cv(pos,size,9,-10)+cv(pos,size,9,-4)+cv(pos,size,9,-3)+cv(pos,size,9,-2)+cv(pos,size,9,2)+cv(pos,size,9,3)+cv(pos,size,9,4)+cv(pos,size,9,10)+
        cv(pos,size,10,-9)+cv(pos,size,10,-1)+cv(pos,size,10,0)+cv(pos,size,10,1)+cv(pos,size,10,9)+
        cv(pos,size,11,-8)+cv(pos,size,11,-7)+cv(pos,size,11,7)+cv(pos,size,11,8)+
        cv(pos,size,12,-6)+cv(pos,size,12,-5)+cv(pos,size,12,5)+cv(pos,size,12,6)+
        cv(pos,size,13,-4)+cv(pos,size,13,-3)+cv(pos,size,13,-2)+cv(pos,size,13,2)+cv(pos,size,13,3)+cv(pos,size,13,4)+
        cv(pos,size,14,-1)+cv(pos,size,14,0)+cv(pos,size,14,1);
    
    // Neighborhood 2: Inner ring at radius ~3
    float nhd2 = 
        cv(pos,size,-3,-1)+cv(pos,size,-3,0)+cv(pos,size,-3,1)+
        cv(pos,size,-2,-2)+cv(pos,size,-2,2)+
        cv(pos,size,-1,-3)+cv(pos,size,-1,-1)+cv(pos,size,-1,0)+cv(pos,size,-1,1)+cv(pos,size,-1,3)+
        cv(pos,size,0,-3)+cv(pos,size,0,-1)+cv(pos,size,0,1)+cv(pos,size,0,3)+
        cv(pos,size,1,-3)+cv(pos,size,1,-1)+cv(pos,size,1,0)+cv(pos,size,1,1)+cv(pos,size,1,3)+
        cv(pos,size,2,-2)+cv(pos,size,2,2)+
        cv(pos,size,3,-1)+cv(pos,size,3,0)+cv(pos,size,3,1);
    
    // Neighborhood 3+4+5: Filled disc pattern
    float nhd3 = 
        cv(pos,size,-14,-3)+cv(pos,size,-14,-2)+cv(pos,size,-14,-1)+cv(pos,size,-14,0)+cv(pos,size,-14,1)+cv(pos,size,-14,2)+cv(pos,size,-14,3)+
        cv(pos,size,-13,-6)+cv(pos,size,-13,-5)+cv(pos,size,-13,-4)+cv(pos,size,-13,-3)+cv(pos,size,-13,-2)+cv(pos,size,-13,-1)+cv(pos,size,-13,0)+cv(pos,size,-13,1)+cv(pos,size,-13,2)+cv(pos,size,-13,3)+cv(pos,size,-13,4)+cv(pos,size,-13,5)+cv(pos,size,-13,6)+
        cv(pos,size,-12,-8)+cv(pos,size,-12,-7)+cv(pos,size,-12,-6)+cv(pos,size,-12,-5)+cv(pos,size,-12,-4)+cv(pos,size,-12,-3)+cv(pos,size,-12,-2)+cv(pos,size,-12,-1)+cv(pos,size,-12,0)+cv(pos,size,-12,1)+cv(pos,size,-12,2)+cv(pos,size,-12,3)+cv(pos,size,-12,4)+cv(pos,size,-12,5)+cv(pos,size,-12,6)+cv(pos,size,-12,7)+cv(pos,size,-12,8)+
        cv(pos,size,-11,-9)+cv(pos,size,-11,-8)+cv(pos,size,-11,-7)+cv(pos,size,-11,-6)+cv(pos,size,-11,-5)+cv(pos,size,-11,-4)+cv(pos,size,-11,-3)+cv(pos,size,-11,-2)+cv(pos,size,-11,-1)+cv(pos,size,-11,0)+cv(pos,size,-11,1)+cv(pos,size,-11,2)+cv(pos,size,-11,3)+cv(pos,size,-11,4)+cv(pos,size,-11,5)+cv(pos,size,-11,6)+cv(pos,size,-11,7)+cv(pos,size,-11,8)+cv(pos,size,-11,9)+
        cv(pos,size,-10,-10)+cv(pos,size,-10,-9)+cv(pos,size,-10,-8)+cv(pos,size,-10,-7)+cv(pos,size,-10,-6)+cv(pos,size,-10,-5)+cv(pos,size,-10,5)+cv(pos,size,-10,6)+cv(pos,size,-10,7)+cv(pos,size,-10,8)+cv(pos,size,-10,9)+cv(pos,size,-10,10)+
        cv(pos,size,-9,-11)+cv(pos,size,-9,-10)+cv(pos,size,-9,-9)+cv(pos,size,-9,-8)+cv(pos,size,-9,-7)+cv(pos,size,-9,7)+cv(pos,size,-9,8)+cv(pos,size,-9,9)+cv(pos,size,-9,10)+cv(pos,size,-9,11)+
        cv(pos,size,-8,-12)+cv(pos,size,-8,-11)+cv(pos,size,-8,-10)+cv(pos,size,-8,-9)+cv(pos,size,-8,-8)+cv(pos,size,-8,8)+cv(pos,size,-8,9)+cv(pos,size,-8,10)+cv(pos,size,-8,11)+cv(pos,size,-8,12)+
        cv(pos,size,-7,-12)+cv(pos,size,-7,-11)+cv(pos,size,-7,-10)+cv(pos,size,-7,-9)+cv(pos,size,-7,-2)+cv(pos,size,-7,-1)+cv(pos,size,-7,0)+cv(pos,size,-7,1)+cv(pos,size,-7,2)+cv(pos,size,-7,9)+cv(pos,size,-7,10)+cv(pos,size,-7,11)+cv(pos,size,-7,12)+
        cv(pos,size,-6,-13)+cv(pos,size,-6,-12)+cv(pos,size,-6,-11)+cv(pos,size,-6,-10)+cv(pos,size,-6,-4)+cv(pos,size,-6,-3)+cv(pos,size,-6,3)+cv(pos,size,-6,4)+cv(pos,size,-6,10)+cv(pos,size,-6,11)+cv(pos,size,-6,12)+cv(pos,size,-6,13)+
        cv(pos,size,-5,-13)+cv(pos,size,-5,-12)+cv(pos,size,-5,-11)+cv(pos,size,-5,-10)+cv(pos,size,-5,-5)+cv(pos,size,-5,5)+cv(pos,size,-5,10)+cv(pos,size,-5,11)+cv(pos,size,-5,12)+cv(pos,size,-5,13)+
        cv(pos,size,-4,-13)+cv(pos,size,-4,-12)+cv(pos,size,-4,-11)+cv(pos,size,-4,-6)+cv(pos,size,-4,-1)+cv(pos,size,-4,0)+cv(pos,size,-4,1)+cv(pos,size,-4,6)+cv(pos,size,-4,11)+cv(pos,size,-4,12)+cv(pos,size,-4,13)+
        cv(pos,size,-3,-14)+cv(pos,size,-3,-13)+cv(pos,size,-3,-12)+cv(pos,size,-3,-11)+cv(pos,size,-3,-6)+cv(pos,size,-3,-2)+cv(pos,size,-3,2)+cv(pos,size,-3,6)+cv(pos,size,-3,11)+cv(pos,size,-3,12)+cv(pos,size,-3,13)+cv(pos,size,-3,14)+
        cv(pos,size,-2,-14)+cv(pos,size,-2,-13)+cv(pos,size,-2,-12)+cv(pos,size,-2,-11)+cv(pos,size,-2,-7)+cv(pos,size,-2,-3)+cv(pos,size,-2,3)+cv(pos,size,-2,7)+cv(pos,size,-2,11)+cv(pos,size,-2,12);
    
    float nhd4 = 
        cv(pos,size,-2,13)+cv(pos,size,-2,14)+
        cv(pos,size,-1,-14)+cv(pos,size,-1,-13)+cv(pos,size,-1,-12)+cv(pos,size,-1,-11)+cv(pos,size,-1,-7)+cv(pos,size,-1,-4)+cv(pos,size,-1,-1)+cv(pos,size,-1,0)+cv(pos,size,-1,1)+cv(pos,size,-1,4)+cv(pos,size,-1,7)+cv(pos,size,-1,11)+cv(pos,size,-1,12)+cv(pos,size,-1,13)+cv(pos,size,-1,14)+
        cv(pos,size,0,-14)+cv(pos,size,0,-13)+cv(pos,size,0,-12)+cv(pos,size,0,-11)+cv(pos,size,0,-7)+cv(pos,size,0,-4)+cv(pos,size,0,-1)+cv(pos,size,0,1)+cv(pos,size,0,4)+cv(pos,size,0,7)+cv(pos,size,0,11)+cv(pos,size,0,12)+cv(pos,size,0,13)+cv(pos,size,0,14)+
        cv(pos,size,1,-14)+cv(pos,size,1,-13)+cv(pos,size,1,-12)+cv(pos,size,1,-11)+cv(pos,size,1,-7)+cv(pos,size,1,-4)+cv(pos,size,1,-1)+cv(pos,size,1,0)+cv(pos,size,1,1)+cv(pos,size,1,4)+cv(pos,size,1,7)+cv(pos,size,1,11)+cv(pos,size,1,12)+cv(pos,size,1,13)+cv(pos,size,1,14)+
        cv(pos,size,2,-14)+cv(pos,size,2,-13)+cv(pos,size,2,-12)+cv(pos,size,2,-11)+cv(pos,size,2,-7)+cv(pos,size,2,-3)+cv(pos,size,2,3)+cv(pos,size,2,7)+cv(pos,size,2,11)+cv(pos,size,2,12)+cv(pos,size,2,13)+cv(pos,size,2,14)+
        cv(pos,size,3,-14)+cv(pos,size,3,-13)+cv(pos,size,3,-12)+cv(pos,size,3,-11)+cv(pos,size,3,-6)+cv(pos,size,3,-2)+cv(pos,size,3,2)+cv(pos,size,3,6)+cv(pos,size,3,11)+cv(pos,size,3,12)+cv(pos,size,3,13)+cv(pos,size,3,14)+
        cv(pos,size,4,-13)+cv(pos,size,4,-12)+cv(pos,size,4,-11)+cv(pos,size,4,-6)+cv(pos,size,4,-1)+cv(pos,size,4,0)+cv(pos,size,4,1)+cv(pos,size,4,6)+cv(pos,size,4,11)+cv(pos,size,4,12)+cv(pos,size,4,13)+
        cv(pos,size,5,-13)+cv(pos,size,5,-12)+cv(pos,size,5,-11)+cv(pos,size,5,-10)+cv(pos,size,5,-5)+cv(pos,size,5,5)+cv(pos,size,5,10)+cv(pos,size,5,11)+cv(pos,size,5,12)+cv(pos,size,5,13)+
        cv(pos,size,6,-13)+cv(pos,size,6,-12)+cv(pos,size,6,-11)+cv(pos,size,6,-10)+cv(pos,size,6,-4)+cv(pos,size,6,-3)+cv(pos,size,6,3)+cv(pos,size,6,4)+cv(pos,size,6,10)+cv(pos,size,6,11)+cv(pos,size,6,12)+cv(pos,size,6,13)+
        cv(pos,size,7,-12)+cv(pos,size,7,-11)+cv(pos,size,7,-10)+cv(pos,size,7,-9)+cv(pos,size,7,-2)+cv(pos,size,7,-1)+cv(pos,size,7,0)+cv(pos,size,7,1)+cv(pos,size,7,2)+cv(pos,size,7,9)+cv(pos,size,7,10)+cv(pos,size,7,11)+cv(pos,size,7,12)+
        cv(pos,size,8,-12)+cv(pos,size,8,-11)+cv(pos,size,8,-10)+cv(pos,size,8,-9)+cv(pos,size,8,-8)+cv(pos,size,8,8)+cv(pos,size,8,9)+cv(pos,size,8,10)+cv(pos,size,8,11)+cv(pos,size,8,12)+
        cv(pos,size,9,-11)+cv(pos,size,9,-10)+cv(pos,size,9,-9)+cv(pos,size,9,-8)+cv(pos,size,9,-7)+cv(pos,size,9,7)+cv(pos,size,9,8)+cv(pos,size,9,9)+cv(pos,size,9,10)+cv(pos,size,9,11)+
        cv(pos,size,10,-10)+cv(pos,size,10,-9)+cv(pos,size,10,-8)+cv(pos,size,10,-7)+cv(pos,size,10,-6)+cv(pos,size,10,-5)+cv(pos,size,10,5)+cv(pos,size,10,6)+cv(pos,size,10,7)+cv(pos,size,10,8)+cv(pos,size,10,9)+cv(pos,size,10,10)+
        cv(pos,size,11,-9)+cv(pos,size,11,-8)+cv(pos,size,11,-7)+cv(pos,size,11,-6)+cv(pos,size,11,-5)+cv(pos,size,11,-4)+cv(pos,size,11,-3)+cv(pos,size,11,-2);
    
    float nhd5 = 
        cv(pos,size,11,-1)+cv(pos,size,11,0)+cv(pos,size,11,1)+cv(pos,size,11,2)+cv(pos,size,11,3)+cv(pos,size,11,4)+cv(pos,size,11,5)+cv(pos,size,11,6)+cv(pos,size,11,7)+cv(pos,size,11,8)+cv(pos,size,11,9)+
        cv(pos,size,12,-8)+cv(pos,size,12,-7)+cv(pos,size,12,-6)+cv(pos,size,12,-5)+cv(pos,size,12,-4)+cv(pos,size,12,-3)+cv(pos,size,12,-2)+cv(pos,size,12,-1)+cv(pos,size,12,0)+cv(pos,size,12,1)+cv(pos,size,12,2)+cv(pos,size,12,3)+cv(pos,size,12,4)+cv(pos,size,12,5)+cv(pos,size,12,6)+cv(pos,size,12,7)+cv(pos,size,12,8)+
        cv(pos,size,13,-6)+cv(pos,size,13,-5)+cv(pos,size,13,-4)+cv(pos,size,13,-3)+cv(pos,size,13,-2)+cv(pos,size,13,-1)+cv(pos,size,13,0)+cv(pos,size,13,1)+cv(pos,size,13,2)+cv(pos,size,13,3)+cv(pos,size,13,4)+cv(pos,size,13,5)+cv(pos,size,13,6)+
        cv(pos,size,14,-3)+cv(pos,size,14,-2)+cv(pos,size,14,-1)+cv(pos,size,14,0)+cv(pos,size,14,1)+cv(pos,size,14,2)+cv(pos,size,14,3);
    
    // Combine neighborhoods
    float fin_0 = nhd0 + nhd1;
    float fin_1 = nhd2;
    float fin_2 = nhd3 + nhd4 + nhd5;
    
    // Rule 388 conditions
    if (fin_0 >= 31.0 && fin_0 <= 155.0) {
        outval = 0.0;
    }
    if (fin_0 >= 125.0 && fin_0 <= 135.0) {
        outval = 0.0;
    }
    if (fin_0 >= 31.0 && fin_0 <= 45.0) {
        outval = 0.0;
    }
    if (fin_0 >= 40.0 && fin_0 <= 42.0) {
        outval = 1.0;
    }
    if (fin_0 >= 87.0 && fin_0 <= 137.0) {
        outval = 1.0;
    }
    if (fin_1 >= 13.0 && fin_1 <= 19.0) {
        outval = 1.0;
    }
    if (fin_1 >= 9.0 && fin_1 <= 9.0) {
        outval = 1.0;
    }
    if (fin_2 >= 185.0) {
        outval = 0.0;
    }
    
    return outval;
}

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(imgState);
    
    if (pos.x >= size.x || pos.y >= size.y) return;
    
    if (pass == 0) {
        // Run Rule 388 cellular automaton
        float newVal = doCellular388(pos, size);
        imageStore(imgStateOut, pos, vec4(newVal, newVal, newVal, 1.0));
    } else {
        // Pass 1: copy
        imageStore(imgStateOut, pos, imageLoad(imgState, pos));
    }
}
