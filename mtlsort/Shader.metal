//
//  Shader.metal
//  MetalSort
//
//  Created by Tim Gymnich on 13.09.17.
//  Copyright © 2017 Tim Gymnich. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

typedef float DataType;

kernel void parallelBitonic(device DataType *input [[buffer(0)]],
                            constant uint &p [[buffer(1)]],
                            constant uint &q [[buffer(2)]],
                            uint tid [[ thread_index_in_threadgroup ]],
                            uint bid [[ threadgroup_position_in_grid ]],
                            uint blockDim [[ threads_per_threadgroup ]])
{

    uint d = 1 << (p - q);

    uint i = bid * blockDim + tid;

    bool up = ((i >> p) & 2) == 0;
    uint id = i | d;
    DataType iVal = input[i];
    DataType idVal = input[id];

    if (((i & d) == 0) && ((bool)(iVal > idVal) == up)) {
        input[i] = idVal;
        input[id] = iVal;
    }
}



