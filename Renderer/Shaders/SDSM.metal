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

kernel void add_arrays(device atomic_int* result,
                       texture2d<half, access::read> inTexture [[texture(0)]],
                       uint2 gid [[thread_position_in_grid]])
{
    half4 inColor = inTexture.read(gid);
    atomic_fetch_add_explicit(result, 1, memory_order_relaxed);
}
