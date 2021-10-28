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

kernel void reduce_min_max_depth(device atomic_int * result [[buffer(BufferIndexMinMaxDepth)]],
                       texture2d<float> depthBuffer [[texture(TextureIndexDepth)]],
                       uint2 gid [[thread_position_in_grid]])
{
    uint depthInt = (uint)(depthBuffer.read(gid).x * LARGE_INTEGER);

    atomic_fetch_max_explicit(&result[1], as_type<int>(depthInt), memory_order_relaxed);

    if (depthInt > 0) {
        atomic_fetch_min_explicit(result, as_type<int>(depthInt), memory_order_relaxed);
    }

    atomic_fetch_add_explicit(&result[2], 1, memory_order_relaxed);
}
