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

// Per-vertex inputs fed by vertex buffer laid out with MTLVertexDescriptor in Metal API
struct DescriptorDefinedVertex
{
    float3 position  [[attribute(VertexAttributePosition)]];
    float2 tex_coord [[attribute(VertexAttributeTexcoord)]];
    half3 normal     [[attribute(VertexAttributeNormal)]];
    half3 tangent    [[attribute(VertexAttributeTangent)]];
    half3 bitangent  [[attribute(VertexAttributeBitangent)]];
};

// Vertex shader outputs and per-fragment inputs.  Includes clip-space position and vertex outputs
// interpolated by rasterizer and fed to each fragment generated by clip-space primitives.
struct ColorInOut
{
    float4 position [[position]];
    float2 tex_coord;
    float2 shadow_uv;
    half   shadow_depth;
    float3 eye_position;
    half3  tangent;
    half3  bitangent;
    half3  normal;
};

vertex ColorInOut directionalLightVertex(DescriptorDefinedVertex in  [[ stage_in ]],
                                         constant FrameData &frameData [[ buffer(BufferIndexFrameData) ]])
{
    ColorInOut out;

    float4 model_position = float4(in.position, 1.0);

    out.position = frameData.projection_matrix * frameData.temple_model_matrix * model_position;
    out.tex_coord = in.tex_coord;

    return out;
}

fragment half4 directionalLightFragment(ColorInOut in [[ stage_in ]])
{
    return half4(0.0, 1.0, 0.0, 1.0);
}
