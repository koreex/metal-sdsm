//
//  File.metal
//  DeferredLighting C++
//
//  Created by Koreex on 2021/8/6.
//  Copyright © 2021 Apple. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

#include "AAPLShaderTypes.h"
#include "AAPLShaderCommon.h"

kernel void add_arrays(device const float* inA,
                       device const float* inB,
                       device float* result,
                       uint index [[thread_position_in_grid]])
{
    // the for-loop is replaced with a collection of threads, each of which
    // calls this function.
    result[index] = inA[index] + inB[index];
}
