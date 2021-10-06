//
//  CPPMetalComputePipelineState.cpp
//  CPPMetal
//
//  Created by Koreex on 2021/10/6.
//  Copyright © 2021 Apple. All rights reserved.
//

#include "CPPMetalComputePipelineState.hpp"
#include "CPPMetalLibrary.hpp"
#include "CPPMetalDevice.hpp"
#include "CPPMetalInternalMacros.h"

using namespace MTL;

#pragma mark - CPPMetalComputePipelineState

CPP_METAL_CONSTRUCTOR_IMPLEMENTATION(ComputePipelineState);

CPP_METAL_NULL_REFERENCE_CONSTRUCTOR_IMPLEMENATATION(ComputePipelineState);

CPP_METAL_COPY_CONSTRUCTOR_AND_OPERATOR_OVERLOAD_IMPLEMENTATION(ComputePipelineState);

ComputePipelineState::~ComputePipelineState()
{
    m_objCObj = nil;
}

const char * ComputePipelineState::label() const
{
    CPP_METAL_VALIDATE_WRAPPED_NIL();

    return m_objCObj.label.UTF8String;
}

bool ComputePipelineState::operator==(const ComputePipelineState & rhs) const
{
    return [m_objCObj isEqual:rhs.objCObj()];
}

CPP_METAL_DEVICE_GETTER_IMPLEMENTATION(ComputePipelineState);
