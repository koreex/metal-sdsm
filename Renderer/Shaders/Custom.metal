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
                       texture2d<half, access::read> inTexture [[texture(0)]],
                       uint2 gid [[thread_position_in_grid]])
{
    half4 inColor = inTexture.read(gid);
    result[0] = inColor.r;
}
