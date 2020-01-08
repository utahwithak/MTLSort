//
//  Shader.metal
//  MetalSort
//
//  Created by Tim Gymnich on 13.09.17.
//  Copyright © 2017 Tim Gymnich. All rights reserved.
//

#include <metal_stdlib>
#include "Structs.h"

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


bool isOrderedBefore(Vector2 a, Vector2 b) {
    if (a.x < b.x) {
        return false;
    } else if (a.x > b.x) {
        return true;
    } else {
        return a.y >= b.y;
    }
}

kernel void pointBitonic(device Vector2 *input [[buffer(0)]],
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
    Vector2 iVal = input[i];
    Vector2 idVal = input[id];

    if (((i & d) == 0) && (isOrderedBefore(iVal, idVal) == up)) {
        input[i] = idVal;
        input[id] = iVal;
    }
}

kernel void boundingBox(const device Vector2 *array [[ buffer(0) ]],
                        volatile device atomic_int *result [[ buffer(1) ]],
                        threadgroup int *shared_X_memory [[threadgroup(0)]],
                        threadgroup int *shared_Y_memory [[threadgroup(1)]],
                        uint id [[ thread_position_in_grid ]],
                        uint tid [[ thread_index_in_threadgroup ]],
                        uint bid [[ threadgroup_position_in_grid ]],
                        uint blockDim [[ threads_per_threadgroup ]]) {


    uint i = bid * blockDim + tid;
    shared_X_memory[tid] = i;
    shared_Y_memory[tid] = i;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 1; stride < blockDim; stride *= 2) {
        // only accept increasing multiples of two
        if (tid % (2 * stride) == 0) {
            if ( array[shared_X_memory[tid + stride]].x < array[shared_X_memory[tid]].x) {
                shared_X_memory[tid] = shared_X_memory[tid + stride];
            }
            if ( array[shared_Y_memory[tid + stride]].y < array[shared_Y_memory[tid]].y) {
                shared_Y_memory[tid] = shared_Y_memory[tid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (stride > 1 && tid % (2 * stride) == 1) {
            // Max calculations
            if (array[shared_X_memory[tid + stride]].x > array[shared_X_memory[tid]].x) {
                shared_X_memory[tid] = shared_X_memory[tid + stride];
            }

            if ( array[shared_Y_memory[tid + stride]].y > array[shared_Y_memory[tid]].y) {
                shared_Y_memory[tid] = shared_Y_memory[tid + stride];
            }

        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (0 == tid ) {
        int currentXMin = atomic_load_explicit(&result[0], memory_order_relaxed);

        if( currentXMin == -1 || array[shared_X_memory[tid]].x < array[currentXMin].x) {
            while(!atomic_compare_exchange_weak_explicit(&result[0], &currentXMin, shared_X_memory[tid], memory_order_relaxed, memory_order_relaxed)) {
                // another thread won. Check if we still need to set it.
                if (array[shared_X_memory[tid]].x > array[currentXMin].x) {
                    break;
                }
            }
        }
        int currentYMin = atomic_load_explicit(&result[1], memory_order_relaxed);

        if( currentYMin == -1 ||  array[shared_Y_memory[tid]].y < array[currentYMin].y) {
            while(!atomic_compare_exchange_weak_explicit(&result[1], &currentYMin, shared_Y_memory[tid], memory_order_relaxed, memory_order_relaxed)) {
                // another thread won. Check if we still need to set it.
                if (array[shared_Y_memory[tid]].y > array[currentYMin].y) {
                    break;
                }
            }
        }
    } else if (tid == 1) {

        int currentXMaxIndex = atomic_load_explicit(&result[2], memory_order_relaxed);

        if(currentXMaxIndex == -1 || array[shared_X_memory[tid]].x > array[currentXMaxIndex].x) {
            while(!atomic_compare_exchange_weak_explicit(&result[2], &currentXMaxIndex, shared_X_memory[tid], memory_order_relaxed, memory_order_relaxed)) {
                // another thread won. Check if we still need to set it.
                if (array[shared_X_memory[tid]].x < array[currentXMaxIndex].x) {
                    break;
                }
            }
        }

        int currentYMax = atomic_load_explicit(&result[3], memory_order_relaxed);

        if(currentYMax == -1 || array[shared_Y_memory[tid]].y > array[currentYMax].y) {
            while(!atomic_compare_exchange_weak_explicit(&result[3], &currentYMax, shared_Y_memory[tid], memory_order_relaxed, memory_order_relaxed)) {
                // another thread won. Check if we still need to set it.
                if (array[shared_Y_memory[tid]].y < array[currentYMax].y) {
                    break;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

}


