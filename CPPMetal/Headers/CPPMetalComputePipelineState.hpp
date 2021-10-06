//
//  CPPMetalComputePipelineState.hpp
//  CPPMetal
//
//  Created by Koreex on 2021/10/6.
//  Copyright © 2021 Apple. All rights reserved.
//

#ifndef CPPMetalComputePipelineState_hpp
#define CPPMetalComputePipelineState_hpp

#include <CoreFoundation/CoreFoundation.h>
#include "CPPMetalImplementation.hpp"
#include "CPPMetalPixelFormat.hpp"
#include "CPPMetalConstants.hpp"
#include "CPPMetalDevice.hpp"
#include "CPPMetalVertexDescriptor.hpp"
#include "CPPMetalLibrary.hpp"
#include "CPPMetalTypes.hpp"


namespace MTL
{

class Device;

class ComputePipelineState
{
public:

    ComputePipelineState();

    ComputePipelineState(const ComputePipelineState & rhs);

    ComputePipelineState(ComputePipelineState && rhs);

    ComputePipelineState & operator=(const ComputePipelineState & rhs);

    ComputePipelineState & operator=(ComputePipelineState && rhs);

    CPP_METAL_VIRTUAL ~ComputePipelineState();

    bool operator==(const ComputePipelineState &rhs) const;

    const char *label() const;

    Device device() const;

private:

    CPPMetalInternal::ComputePipelineState m_objCObj;

    Device *m_device;

public: // Public methods for CPPMetal internal implementation

    ComputePipelineState(CPPMetalInternal::ComputePipelineState objCObj, Device &device);

    CPPMetalInternal::ComputePipelineState objCObj() const;
};

}

#endif /* CPPMetalComputePipelineState_hpp */
