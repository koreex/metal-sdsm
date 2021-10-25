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

kernel void reduce_min_max_depth(device atomic_int * result [[buffer(0)]],
                       texture2d<float> depthBuffer [[texture(0)]],
                       uint2 gid [[thread_position_in_grid]])
{
    uint depthInt = (uint)(depthBuffer.read(gid).x * 1000);

    uint maxDepth = atomic_load_explicit(result, memory_order_relaxed);
    uint minDepth = atomic_load_explicit(&result[1], memory_order_relaxed);

    if (depthInt > maxDepth) {
        atomic_exchange_explicit(result, depthInt, memory_order_relaxed);
    }

    if (depthInt > 0 && depthInt < minDepth) {
        atomic_exchange_explicit(&result[1], depthInt, memory_order_relaxed);
    }

    atomic_fetch_add_explicit(&result[2], 1, memory_order_relaxed);
}
