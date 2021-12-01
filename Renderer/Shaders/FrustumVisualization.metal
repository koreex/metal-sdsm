//
//  FrustumVisualization.metal
//  DeferredLighting C++
//
//  Created by Koreex on 2021/11/26.
//  Copyright © 2021 Apple. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

#include "AAPLShaderTypes.h"

struct VertexOut
{
    float4 position [[ position ]];
    float3 color;
};

vertex VertexOut frustum_vertex(
                                uint vertexId [[ vertex_id ]],
                                constant FrustumVertex *vertices [[ buffer(0) ]],
                                constant FrameData &frameData [[ buffer(1) ]])
{
    VertexOut out;
    out.position = frameData.projection_matrix * frameData.view_matrix *
        float4(vertices[vertexId].position.xyz, 1.0f);
    out.color = vertices[vertexId].color;

    return out;
}

fragment float4 frustum_fragment(VertexOut in [[stage_in]])
{
    return vector_float4(in.color, 1.0f);
}
