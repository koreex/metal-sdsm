//
//  CPPMetalComputeCommandEncoder.m
//  CPPMetal
//
//  Created by Koreex on 2021/10/8.
//  Copyright © 2021 Apple. All rights reserved.
//

#include "CPPMetalDevice.hpp"
#include "CPPMetalComputeCommandEncoder.hpp"
#include "CPPMetalInternalMacros.h"
#include "CPPMetalDepthStencil.hpp"
#include "CPPMetalDeviceInternals.h"
#include <Metal/Metal.h>

using namespace MTL;


ComputeCommandEncoder::ComputeCommandEncoder(const CPPMetalInternal::ComputeCommandEncoder objCObj,
                                           Device & device)
: CommandEncoder(objCObj, device)
{
    m_dispatch = m_device->internals().getComputeCommandEncoderTable(objCObj);
}

ComputeCommandEncoder::ComputeCommandEncoder(const ComputeCommandEncoder & rhs)
: CommandEncoder(rhs)
, m_dispatch(rhs.m_dispatch)
{
    // Member initialization only
}

ComputeCommandEncoder & ComputeCommandEncoder::operator=(const ComputeCommandEncoder & rhs)
{
    CommandEncoder::operator=(rhs);

    return *this;
}

ComputeCommandEncoder::~ComputeCommandEncoder()
{
}


bool ComputeCommandEncoder::operator==(const ComputeCommandEncoder & rhs) const
{
    return [((id<MTLComputeCommandEncoder>)m_objCObj) isEqual:rhs.m_objCObj];
}
