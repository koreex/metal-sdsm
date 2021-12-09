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
}

kernel void reduce_light_frustum(texture2d<float> depthBuffer [[texture(TextureIndexDepth)]],
                                constant     FrameData    &frameData [[ buffer(BufferIndexFrameData) ]],
                                uint2 gid [[thread_position_in_grid]],
                                device atomic_int * lightFrustumBoundingBox [[ buffer(BufferIndexBoundingBox) ]])
{
    float depth = depthBuffer.read(uint2(gid.x, gid.y)).x;
    if (depth < 1e-6) {
        return;
    }

    // convert to nonlinear depth;
    float4 samplePosition = frameData.projection_matrix * float4(0, 0, depth, 1.0);

    float4 positionLS = frameData.unproject_matrix * float4(gid.x, gid.y, samplePosition.z / samplePosition.w, 1.0);
    positionLS /= positionLS.w;
    positionLS = frameData.shadow_view_matrix * positionLS;

    for (uint i = 0; i < CASCADED_SHADOW_COUNT; i++) {
        if (depth > frameData.cascadeEnds[i] && depth < frameData.cascadeEnds[i + 1]) {
            atomic_fetch_min_explicit(&lightFrustumBoundingBox[6 * i + BoundingBoxMinX],
                                      as_type<int>((int)(positionLS.x * LARGE_INTEGER)),
                                      memory_order_relaxed);
            atomic_fetch_min_explicit(&lightFrustumBoundingBox[6 * i + BoundingBoxMinY],
                                      as_type<int>((int)(positionLS.y * LARGE_INTEGER)),
                                      memory_order_relaxed);
            atomic_fetch_min_explicit(&lightFrustumBoundingBox[6 * i + BoundingBoxMinZ],
                                      as_type<int>((int)(positionLS.z * LARGE_INTEGER)),
                                      memory_order_relaxed);

            atomic_fetch_max_explicit(&lightFrustumBoundingBox[6 * i + BoundingBoxMaxX],
                                      as_type<int>((int)(positionLS.x * LARGE_INTEGER)),
                                      memory_order_relaxed);
            atomic_fetch_max_explicit(&lightFrustumBoundingBox[6 * i + BoundingBoxMaxY],
                                      as_type<int>((int)(positionLS.y * LARGE_INTEGER)),
                                      memory_order_relaxed);
            atomic_fetch_max_explicit(&lightFrustumBoundingBox[6 * i + BoundingBoxMaxZ],
                                      as_type<int>((int)(positionLS.z * LARGE_INTEGER)),
                                      memory_order_relaxed);
        }
    }
}
