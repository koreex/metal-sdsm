//
//  Header.h
//  DeferredLighting C++
//
//  Created by Koreex on 2021/10/9.
//  Copyright © 2021 Apple. All rights reserved.
//

#ifndef CPPMetalComputeCommandEncoder_DispatchTable_hpp
#define CPPMetalComputeCommandEncoder_DispatchTable_hpp

#include "CPPMetalBuffer.hpp"
#include "CPPMetalComputePipeline.hpp"
#include "CPPMetalTexture.hpp"
#include "CPPMetalTypes.hpp"
#include "CPPMetalImplementation.hpp"
#include <objc/message.h>


namespace CPPMetalInternal
{

static const SEL setBufferSel          = sel_registerName("setBuffer:offset:atIndex:");
static const SEL setTextureSel = sel_registerName("setTexture:atIndex:");

typedef void (*setBufferType)         (id, SEL, CPPMetalInternal::Buffer buffer, MTL::UInteger offset, MTL::UInteger index);
typedef void (*setTextureType)(id, SEL, CPPMetalInternal::Texture texture, MTL::UInteger index);

struct ComputeCommandEncoderDispatchTable
{
    CPP_METAL_DECLARE_FUNCTION_POINTER( setBuffer );
    CPP_METAL_DECLARE_FUNCTION_POINTER( setTexture );

    ComputeCommandEncoderDispatchTable(CPPMetalInternal::ObjCObj *objCObj);
};

} // namespace CPPMetalInternal

#endif /* CPPMetalComputeCommandEncoder_DispatchTable_hpp */
