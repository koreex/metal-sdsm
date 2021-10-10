/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Header for internal class encapsulating dispatch tables and the allocator
*/

#ifndef CPPMetalDeviceImplementation_h
#define CPPMetalDeviceImplementation_h

#include <Metal/Metal.h>
#include "CPPMetal.hpp"
#include "CPPMetalAllocator.hpp"
#include "CPPMetalRenderCommandEncoder_DispatchTable.hpp"
#include "CPPMetalComputeCommandEncoder_DispatchTable.hpp"
#include "CPPMetalDispatchTableCache.h"


using namespace MTL;

namespace CPPMetalInternal
{

class DeviceInternals
{
public:

    DeviceInternals(Allocator *allocator);

    CPP_METAL_VIRTUAL ~DeviceInternals();

    Allocator & allocator();

    RenderCommandEncoderDispatchTable* getRenderCommandEncoderTable(id<MTLRenderCommandEncoder> objCObj);
    ComputeCommandEncoderDispatchTable* getComputeCommandEncoderTable(id<MTLComputeCommandEncoder> objCObj);

private:

    Allocator *m_allocator;

    CPPMetalInternal::DispatchTableCache<RenderCommandEncoderDispatchTable> m_renderCommandEncoderCache;
    CPPMetalInternal::DispatchTableCache<ComputeCommandEncoderDispatchTable> m_computeCommandEncoderCache;

};

inline Allocator & DeviceInternals::allocator()
{
    return *m_allocator;
}


inline RenderCommandEncoderDispatchTable* DeviceInternals::getRenderCommandEncoderTable(id<MTLRenderCommandEncoder> objCObj)
{
    return m_renderCommandEncoderCache.getTable(objCObj);
}

inline ComputeCommandEncoderDispatchTable* DeviceInternals::getComputeCommandEncoderTable(id<MTLComputeCommandEncoder> objCObj)
{
    return m_computeCommandEncoderCache.getTable(objCObj);
}

} // namespace CPPMetalInternal

#endif // CPPMetalDeviceImplementation_h
