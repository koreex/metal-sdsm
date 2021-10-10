//
//  CPPMetalComputeCommandEncoder.hpp
//  DeferredLighting C++
//
//  Created by Koreex on 2021/10/8.
//  Copyright © 2021 Apple. All rights reserved.
//

#ifndef CPPMetalComputeCommandEncoder_h
#define CPPMetalComputeCommandEncoder_h

#include "CPPMetalCommandEncoder.hpp"
#include "CPPMetalComputeCommandEncoder_DispatchTable.hpp"


namespace MTL
{

class ComputeCommandEncoder : public CommandEncoder
{
public:

    ComputeCommandEncoder() = delete;

    ComputeCommandEncoder(const ComputeCommandEncoder &rhs);

    ComputeCommandEncoder(ComputeCommandEncoder &&rhs);

    ComputeCommandEncoder &operator=(const ComputeCommandEncoder &rhs);

    ComputeCommandEncoder &operator=(ComputeCommandEncoder &&rhs);

    CPP_METAL_VIRTUAL ~ComputeCommandEncoder();

    bool operator==(const ComputeCommandEncoder &rhs) const;

    // Compute State

    void setComputePipelineState(const ComputePipelineState & pipelineState);

    void setBuffer(const Buffer &buffer, UInteger offset, UInteger index);
private:

    CPPMetalInternal::ComputeCommandEncoderDispatchTable *m_dispatch;

public: // Public methods for CPPMetal internal implementation

    ComputeCommandEncoder(const CPPMetalInternal::ComputeCommandEncoder objCObj, Device & device);
};

}

#endif /* CPPMetalComputeCommandEncoder_h */
