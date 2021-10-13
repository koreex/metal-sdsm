//
//  CPPMetalComputeCommandEncoder_DispatchTable.m
//  CPPMetal
//
//  Created by Koreex on 2021/10/9.
//  Copyright © 2021 Apple. All rights reserved.
//

#include "CPPMetalComputeCommandEncoder_DispatchTable.hpp"
#import <objc/runtime.h>

using namespace CPPMetalInternal;

#define CPP_METAL_SET_IMPLEMENTATION(methodName) \
    methodName = (methodName ## Type)[objCObj methodForSelector:methodName ## Sel]

ComputeCommandEncoderDispatchTable::ComputeCommandEncoderDispatchTable(NSObject *objCObj)
{
    CPP_METAL_SET_IMPLEMENTATION( setBuffer );
    CPP_METAL_SET_IMPLEMENTATION( setTexture );
}

