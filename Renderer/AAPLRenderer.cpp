/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Implementation of renderer class which performs Metal setup and per frame rendering
*/

#include "AAPLUtilities.h"

#include<sys/sysctl.h>
#include <simd/simd.h>
#include <stdlib.h>

#include "AAPLBufferExaminationManager.h"
#include "AAPLRenderer.h"
#include "AAPLMesh.h"
#include "AAPLMathUtilities.h"
#include "SDSM_Utilities.h"

using namespace simd;

// Include header shared between C code here, which executes Metal API commands, and .metal files
#include "AAPLShaderTypes.h"

Renderer::Renderer(MTK::View & view)
: m_view(view)
, m_device(view.device())
, m_completedHandler(nullptr)
, m_originalLightPositions(nullptr)
, m_frameDataBufferIndex(0)
, m_frameNumber(0)
#if SUPPORT_BUFFER_EXAMINATION
, m_bufferExaminationManager(nullptr)
#endif
{
    this->m_inFlightSemaphore = dispatch_semaphore_create(MaxFramesInFlight);
    this->m_camera = new Camera();
    this->m_camera->setNear(NearPlane);
    this->m_camera->setFar(FarPlane);
    this->m_lightPhi = 0.0f;
    this->m_lightTheta = 2.0f;
    this->m_partitioningMode = LOG_PARTITIONING;
}


Renderer::~Renderer()
{
    delete [] m_originalLightPositions;

    delete m_meshes;

    delete m_completedHandler;
}

/// Create Metal render state objects
void Renderer::loadMetal()
{
    // Create and load the basic Metal state objects
    CFErrorRef error = nullptr;

    printf("Selected Device: %s\n", m_view.device().name());

    for(uint8_t i = 0; i < MaxFramesInFlight; i++)
    {
        // Indicate shared storage so that both the CPU can access the buffers
        static const MTL::ResourceOptions storageMode = MTL::ResourceStorageModeShared;

        m_uniformBuffers[i] = m_device.makeBuffer(sizeof(FrameData), storageMode);

        m_uniformBuffers[i].label("UniformBuffer");

//        m_lightPositions[i] = m_device.makeBuffer(sizeof(float4)*NumLights, storageMode);

        m_uniformBuffers[i].label("LightPositions");
    }

    MTL::Library shaderLibrary = makeShaderLibrary();

    // Positions.

    m_defaultVertexDescriptor.attributes[VertexAttributePosition].format( MTL::VertexFormatFloat3 );
    m_defaultVertexDescriptor.attributes[VertexAttributePosition].offset( 0 );
    m_defaultVertexDescriptor.attributes[VertexAttributePosition].bufferIndex( BufferIndexMeshPositions );

    // Texture coordinates.
    m_defaultVertexDescriptor.attributes[VertexAttributeTexcoord].format( MTL::VertexFormatFloat2 );
    m_defaultVertexDescriptor.attributes[VertexAttributeTexcoord].offset( 0 );
    m_defaultVertexDescriptor.attributes[VertexAttributeTexcoord].bufferIndex( BufferIndexMeshGenerics );

    // Normals.
    m_defaultVertexDescriptor.attributes[VertexAttributeNormal].format( MTL::VertexFormatHalf4 );
    m_defaultVertexDescriptor.attributes[VertexAttributeNormal].offset( 8 );
    m_defaultVertexDescriptor.attributes[VertexAttributeNormal].bufferIndex( BufferIndexMeshGenerics );

    // Tangents
    m_defaultVertexDescriptor.attributes[VertexAttributeTangent].format( MTL::VertexFormatHalf4 );
    m_defaultVertexDescriptor.attributes[VertexAttributeTangent].offset( 16 );
    m_defaultVertexDescriptor.attributes[VertexAttributeTangent].bufferIndex( BufferIndexMeshGenerics );

    // Bitangents
    m_defaultVertexDescriptor.attributes[VertexAttributeBitangent].format( MTL::VertexFormatHalf4 );
    m_defaultVertexDescriptor.attributes[VertexAttributeBitangent].offset( 24 );
    m_defaultVertexDescriptor.attributes[VertexAttributeBitangent].bufferIndex( BufferIndexMeshGenerics );

    // Position Buffer Layout
    m_defaultVertexDescriptor.layouts[BufferIndexMeshPositions].stride( 12 );
    m_defaultVertexDescriptor.layouts[BufferIndexMeshPositions].stepRate( 1 );
    m_defaultVertexDescriptor.layouts[BufferIndexMeshPositions].stepFunction( MTL::VertexStepFunctionPerVertex );

    // Generic Attribute Buffer Layout
    m_defaultVertexDescriptor.layouts[BufferIndexMeshGenerics].stride( 32 );
    m_defaultVertexDescriptor.layouts[BufferIndexMeshGenerics].stepRate( 1 );
    m_defaultVertexDescriptor.layouts[BufferIndexMeshGenerics].stepFunction( MTL::VertexStepFunctionPerVertex );

    m_view.depthStencilPixelFormat( MTL::PixelFormatDepth32Float_Stencil8 );
    m_view.colorPixelFormat( MTL::PixelFormatBGRA8Unorm_sRGB);

    m_albedo_specular_GBufferFormat = MTL::PixelFormatRGBA8Unorm_sRGB;
    m_normal_shadow_GBufferFormat   = MTL::PixelFormatRGBA8Snorm;
    m_depth_GBufferFormat           = MTL::PixelFormatR32Float;

    #pragma mark GBuffer render pipeline setup
    {
        {
            MTL::Function GBufferVertexFunction = shaderLibrary.makeFunction( "gbuffer_vertex" );
            MTL::Function GBufferFragmentFunction = shaderLibrary.makeFunction( "gbuffer_fragment" );

            MTL::RenderPipelineDescriptor renderPipelineDescriptor;

            renderPipelineDescriptor.label( "G-buffer Creation" );
            renderPipelineDescriptor.vertexDescriptor( &m_defaultVertexDescriptor );

            if(m_singlePassDeferred)
            {
                renderPipelineDescriptor.colorAttachments[RenderTargetLighting].pixelFormat( m_view.colorPixelFormat() );
            }
            else
            {
                renderPipelineDescriptor.colorAttachments[RenderTargetLighting].pixelFormat( MTL::PixelFormatInvalid );
            }

            renderPipelineDescriptor.colorAttachments[RenderTargetAlbedo].pixelFormat( m_albedo_specular_GBufferFormat );
            renderPipelineDescriptor.colorAttachments[RenderTargetNormal].pixelFormat( m_normal_shadow_GBufferFormat );
            renderPipelineDescriptor.colorAttachments[RenderTargetDepth].pixelFormat( m_depth_GBufferFormat );
            renderPipelineDescriptor.depthAttachmentPixelFormat( m_view.depthStencilPixelFormat() );
            renderPipelineDescriptor.stencilAttachmentPixelFormat( m_view.depthStencilPixelFormat() );

            renderPipelineDescriptor.vertexFunction( &GBufferVertexFunction );
            renderPipelineDescriptor.fragmentFunction( &GBufferFragmentFunction );

            m_GBufferPipelineState = m_device.makeRenderPipelineState( renderPipelineDescriptor, &error );

            AAPLAssert(error == nullptr, error, "Failed to create GBuffer render pipeline state");
        }

        #pragma mark GBuffer depth state setup
        {
#if LIGHT_STENCIL_CULLING
            MTL::StencilDescriptor stencilStateDesc;
            stencilStateDesc.stencilCompareFunction( MTL::CompareFunctionAlways );
            stencilStateDesc.stencilFailureOperation( MTL::StencilOperationKeep );
            stencilStateDesc.depthFailureOperation( MTL::StencilOperationKeep );
            stencilStateDesc.depthStencilPassOperation( MTL::StencilOperationReplace );
            stencilStateDesc.readMask( 0x0 );
            stencilStateDesc.writeMask( 0xFF );
#else
            MTL::StencilDescriptor stencilStateDesc;
#endif
            MTL::DepthStencilDescriptor depthStencilDesc;
            depthStencilDesc.label( "G-buffer Creation" );
            depthStencilDesc.depthCompareFunction( MTL::CompareFunctionLess );
            depthStencilDesc.depthWriteEnabled( true );
            depthStencilDesc.frontFaceStencil = stencilStateDesc;
            depthStencilDesc.backFaceStencil = stencilStateDesc;

            m_GBufferDepthStencilState = m_device.makeDepthStencilState( depthStencilDesc );
        }
    }

    // Setup render state to apply directional light and shadow in final pass
    {
        #pragma mark Directional lighting render pipeline setup
        {
            MTL::Function directionalVertexFunction = shaderLibrary.makeFunction( "deferred_direction_lighting_vertex" );
            MTL::Function directionalFragmentFunction;

            if(m_singlePassDeferred)
            {
                directionalFragmentFunction =
                    shaderLibrary.makeFunction( "deferred_directional_lighting_fragment_single_pass" );
            }
            else
            {
                directionalFragmentFunction =
                    shaderLibrary.makeFunction( "deferred_directional_lighting_fragment_traditional" );
            }

            MTL::RenderPipelineDescriptor renderPipelineDescriptor;

            renderPipelineDescriptor.label( "Deferred Directional Lighting" );
            renderPipelineDescriptor.vertexDescriptor( nullptr );
            renderPipelineDescriptor.vertexFunction( &directionalVertexFunction );
            renderPipelineDescriptor.fragmentFunction( &directionalFragmentFunction );
            renderPipelineDescriptor.colorAttachments[RenderTargetLighting].pixelFormat( m_view.colorPixelFormat() );

            if(m_singlePassDeferred)
            {
                renderPipelineDescriptor.colorAttachments[RenderTargetAlbedo].pixelFormat( m_albedo_specular_GBufferFormat );
                renderPipelineDescriptor.colorAttachments[RenderTargetNormal].pixelFormat( m_normal_shadow_GBufferFormat );
                renderPipelineDescriptor.colorAttachments[RenderTargetDepth].pixelFormat( m_depth_GBufferFormat );
            }

            renderPipelineDescriptor.depthAttachmentPixelFormat( m_view.depthStencilPixelFormat() );
            renderPipelineDescriptor.stencilAttachmentPixelFormat( m_view.depthStencilPixelFormat() );

            m_directionalLightPipelineState = m_device.makeRenderPipelineState(renderPipelineDescriptor,
                                                                                            &error);

            AAPLAssert(error == nullptr, error,
                       "Failed to create directional light render pipeline state:");
        }

        #pragma mark Directional lighting mask depth stencil state setup
        {
#if LIGHT_STENCIL_CULLING
            // Stencil state setup so direction lighting fragment shader only executed on pixels
            // drawn in GBuffer stage (i.e. mask out the background/sky)
            MTL::StencilDescriptor stencilStateDesc;
            stencilStateDesc.stencilCompareFunction( MTL::CompareFunctionEqual );
            stencilStateDesc.stencilFailureOperation( MTL::StencilOperationKeep );
            stencilStateDesc.depthFailureOperation( MTL::StencilOperationKeep );
            stencilStateDesc.depthStencilPassOperation( MTL::StencilOperationKeep );
            stencilStateDesc.readMask( 0xFF );
            stencilStateDesc.writeMask( 0x0 );
#else
            MTL::StencilDescriptor stencilStateDesc;
#endif
            MTL::DepthStencilDescriptor depthStencilDesc;
            depthStencilDesc.label( "Deferred Directional Lighting" );
            depthStencilDesc.depthWriteEnabled( false );
            depthStencilDesc.depthCompareFunction( MTL::CompareFunctionAlways );
            depthStencilDesc.frontFaceStencil = stencilStateDesc;
            depthStencilDesc.backFaceStencil = stencilStateDesc;

            m_directionLightDepthStencilState = m_device.makeDepthStencilState(depthStencilDesc);
        }
    }

    #pragma mark Post lighting depth state setup
    {
        MTL::DepthStencilDescriptor depthStencilDesc;
        depthStencilDesc.label( "Less -Writes" );
        depthStencilDesc.depthCompareFunction( MTL::CompareFunctionLess );
        depthStencilDesc.depthWriteEnabled( false );

        m_dontWriteDepthStencilState = m_device.newDepthStencilStateWithDescriptor( depthStencilDesc );
    }

    // Setup objects for shadow pass
    {
        MTL::PixelFormat shadowMapPixelFormat = MTL::PixelFormatDepth16Unorm;

        #pragma mark Shadow pass render pipeline setup
        {
            MTL::Function * shadowVertexFunction = shaderLibrary.newFunctionWithName( "shadow_vertex" );

            MTL::RenderPipelineDescriptor renderPipelineDescriptor;
            renderPipelineDescriptor.label( "Shadow Gen" );
            renderPipelineDescriptor.vertexDescriptor( nullptr );
            renderPipelineDescriptor.vertexFunction( shadowVertexFunction );
            renderPipelineDescriptor.fragmentFunction( nullptr );
            renderPipelineDescriptor.depthAttachmentPixelFormat( shadowMapPixelFormat );

            m_shadowGenPipelineState = m_device.makeRenderPipelineState(renderPipelineDescriptor, &error);

            delete shadowVertexFunction;
        }

        #pragma mark Shadow pass depth state setup
        {
            MTL::DepthStencilDescriptor depthStencilDesc;
            depthStencilDesc.label( "Shadow Gen" );
            depthStencilDesc.depthCompareFunction( MTL::CompareFunctionLessEqual );
            depthStencilDesc.depthWriteEnabled( true );
            m_shadowDepthStencilState = m_device.makeDepthStencilState( depthStencilDesc );
        }

        #pragma mark Shadow map setup
        {
            MTL::TextureDescriptor shadowTextureDesc;

            shadowTextureDesc.textureType(MTL::TextureType2DArray);
            shadowTextureDesc.arrayLength(CASCADED_SHADOW_COUNT);
            shadowTextureDesc.pixelFormat( shadowMapPixelFormat );
            shadowTextureDesc.width( SHADOW_MAP_RES );
            shadowTextureDesc.height( SHADOW_MAP_RES );
            shadowTextureDesc.mipmapLevelCount( 1 );
            shadowTextureDesc.resourceOptions( MTL::ResourceStorageModePrivate );
            shadowTextureDesc.usage( MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead );

            m_shadowMap = m_device.makeTexture( shadowTextureDesc );
            m_shadowMap.label( "Shadow Map" );
        }

        #pragma mark Shadow render pass descriptor setup
        {
            m_shadowRenderPassDescriptor.depthAttachment.texture( m_shadowMap );
            m_shadowRenderPassDescriptor.depthAttachment.loadAction( MTL::LoadActionClear );
            m_shadowRenderPassDescriptor.depthAttachment.storeAction( MTL::StoreActionStore );
            m_shadowRenderPassDescriptor.depthAttachment.clearDepth( 1.0 );
            m_shadowRenderPassDescriptor.depthAttachment.slice(0);
        }

        // Create buffers for cascade index
        {
            for (uint i = 0; i < CASCADED_SHADOW_COUNT; i++) {
                static const MTL::ResourceOptions storageMode = MTL::ResourceStorageModeShared;
                m_cascadeIndexBuffers[i] = m_device.makeBuffer(sizeof(int), storageMode);
                int *index = (int*) m_cascadeIndexBuffers[i].contents();
                *index = i;
            }
        }

        #pragma mark Compute pipeline setup
        {
            MTL::Function addArrayFunction = shaderLibrary.makeFunction("reduce_min_max_depth");

            m_reduceComputePipelineState = m_device.makeComputePipelineState(addArrayFunction);

            static const MTL::ResourceOptions storageMode = MTL::ResourceStorageModeShared;

            m_minMaxDepthBuffer = m_device.makeBuffer(sizeof(int) * 2, storageMode);

            int *dataPtrResult = (int*) m_minMaxDepthBuffer.contents();
            dataPtrResult[0] = NearPlane * LARGE_INTEGER;
            dataPtrResult[1] = FarPlane * LARGE_INTEGER;
        }

    }

    m_commandQueue = m_device.makeCommandQueue();
}

/// Load models/textures, etc.
void Renderer::loadScene()
{
    // Create and load assets into Metal objects including meshes and textures
    CFErrorRef error = nullptr;

    m_meshes = newMeshesFromBundlePath("Meshes/Temple.obj", m_device, m_defaultVertexDescriptor, &error);
//    m_meshes = newMeshesFromBundlePath("Meshes/Simple.obj", m_device, m_defaultVertexDescriptor, &error);
//    m_meshes = newMeshesFromBundlePath("Meshes/Hut.obj", m_device, m_defaultVertexDescriptor, &error);

    AAPLAssert(m_meshes, error, "Could not create meshes from model file");

    /**
    // Generate data
    {
        m_lightsData = m_device.makeBuffer(sizeof(PointLight)*NumLights);

        m_lightsData.label ( "LightData" );

        populateLights();
    }

    */

    // Create quad for fullscreen composition drawing
    {
        static const SimpleVertex QuadVertices[] =
        {
            { { -1.0f,  -1.0f, } },
            { { -1.0f,   1.0f, } },
            { {  1.0f,  -1.0f, } },

            { {  1.0f,  -1.0f, } },
            { { -1.0f,   1.0f, } },
            { {  1.0f,   1.0f, } },
        };

        m_quadVertexBuffer = m_device.makeBuffer(QuadVertices, sizeof(QuadVertices));

        m_quadVertexBuffer.label( "Quad Vertices" );
    }

    // Load textures for non mesh assets
    {
        MTK::TextureLoader textureLoader(m_device);

        MTK::TextureLoaderOptions textureLoaderOptions;

        textureLoaderOptions.usage = MTL::TextureUsageShaderRead;
        textureLoaderOptions.storageMode = MTL::StorageModePrivate;

        m_skyMap = textureLoader.makeTexture("SkyMap",
                                             1.0,
                                             textureLoaderOptions,
                                             &error );

        AAPLAssert( error == nullptr, error, "Could not load sky texture" );

        m_skyMap.label( "Sky Map" );

        m_fairyMap = textureLoader.makeTexture("FairyMap",
                                               1.0,
                                               textureLoaderOptions,
                                               &error );

        AAPLAssert( error == nullptr, error, "Could not load fairy texture" );

        m_fairyMap.label( "Fairy Map" );
    }
}

/// Update application state for the current frame
void Renderer::updateWorldState()
{
    if(!m_view.isPaused())
    {
        m_frameNumber++;
    }
    m_frameDataBufferIndex = (m_frameDataBufferIndex+1) % MaxFramesInFlight;

    FrameData *frameData = (FrameData *) (m_uniformBuffers[m_frameDataBufferIndex].contents());

    // Set projection matrix and calculate inverted projection matrix
    frameData->projection_matrix = this->m_camera->projMatrix();
    frameData->projection_matrix_inverse = matrix_invert(this->m_camera->projMatrix());

    // Set screen dimensions
    frameData->framebuffer_width = (uint)m_albedo_specular_GBuffer.width();
    frameData->framebuffer_height = (uint)m_albedo_specular_GBuffer.height();

    frameData->shininess_factor = 1;
    frameData->fairy_specular_intensity = 32;

//    float cameraRotationRadians = m_frameNumber * 0.0025f + M_PI;

//    float4x4 view_matrix = matrix_look_at_left_hand(0,  18, -50,
//                                                    0,   5,   0,
//                                                    0 ,  1,   0);

    float4x4 view_matrix = m_camera->viewMatrix();

    view_matrix = view_matrix * matrix4x4_scale(1, 1, 1);

    frameData->view_matrix = view_matrix;

    float4x4 templeScaleMatrix = matrix4x4_scale(0.1, 0.1, 0.1);
    float4x4 templeTranslateMatrix = matrix4x4_translation(0, -10, 0);
    float4x4 templeModelMatrix = templeTranslateMatrix * templeScaleMatrix;
    frameData->temple_model_matrix = templeModelMatrix;
    frameData->temple_modelview_matrix = frameData->view_matrix * templeModelMatrix;
    frameData->temple_normal_matrix = matrix3x3_upper_left(frameData->temple_modelview_matrix);

//    float skyRotation = m_frameNumber * 0.005f - (M_PI_4*3);

    float3 yAxis = {0, 1, 0}, zAxis = {0, 0, 1};
    float4x4 skyModelMatrix = matrix4x4_rotation(m_lightPhi, yAxis) * matrix4x4_rotation(m_lightTheta, zAxis);
    frameData->sky_modelview_matrix = skyModelMatrix;

    // Update directional light color
    float4 sun_color = {0.5, 0.5, 0.5, 1.0};
    frameData->sun_color = sun_color;
    frameData->sun_specular_intensity = 1;

    // Update sun direction in view space
    float4 sunModelPosition = {0.0, 1.0, 0.0, 0.0};

    float4 sunWorldPosition = skyModelMatrix * sunModelPosition;

    float4 sunWorldDirection = -sunWorldPosition;

    frameData->sun_eye_direction = view_matrix * sunWorldDirection;
    frameData->sun_world_direction = sunWorldDirection;

    // Calculate cascade projection matrices

    {
        float cascadeEnds[CASCADED_SHADOW_COUNT + 1];
        int *dataPtrResult = (int*) m_minMaxDepthBuffer.contents();

        if (this->m_partitioningMode == LOG_PARTITIONING) {
            logPartitioning((float)dataPtrResult[0] / (float)LARGE_INTEGER,
                            (float)dataPtrResult[1] / (float)LARGE_INTEGER,
                            CASCADED_SHADOW_COUNT,
                            cascadeEnds);
        } else {
            unitPartitioning((float)dataPtrResult[0] / (float)LARGE_INTEGER,
                            (float)dataPtrResult[1] / (float)LARGE_INTEGER,
                            CASCADED_SHADOW_COUNT,
                            cascadeEnds);
        }

        FrameData *frameData = (FrameData *) (m_uniformBuffers[m_frameDataBufferIndex].contents());

        for (uint i = 0; i < CASCADED_SHADOW_COUNT + 1; i++) {
            frameData->cascadeEnds[i] = cascadeEnds[i];
        }

        float4 directionalLightUpVector = {0.0, 1.0, 1.0, 1.0};

        directionalLightUpVector = skyModelMatrix * directionalLightUpVector;

        directionalLightUpVector.xyz = normalize(directionalLightUpVector.xyz);

        float4x4 shadowViewMatrix = matrix_look_at_left_hand(sunWorldDirection.xyz / 10,
                                                                    (float3){0,0,0},
                                                                    directionalLightUpVector.xyz);

        float4x4 shadowModelViewMatrix = shadowViewMatrix * templeModelMatrix;

        for (uint i = 0; i < CASCADED_SHADOW_COUNT; i++) {

            float4x4 shadowProjectionMatrix = cascadedShadowProjectionMatrix(
                 this->camera()->viewMatrix(), this->camera()->aspect(), this->camera()->fov(),
                 shadowViewMatrix, cascadeEnds, i);

            frameData->shadow_mvp_matrices[i] = shadowProjectionMatrix * shadowModelViewMatrix;

            // When calculating texture coordinates to sample from shadow map, flip the y/t coordinate and
            // convert from the [-1, 1] range of clip coordinates to [0, 1] range of
            // used for texture sampling
            float4x4 shadowScale = matrix4x4_scale(0.5f, -0.5f, 1.0);
            float4x4 shadowTranslate = matrix4x4_translation(0.5, 0.5, 0);
            float4x4 shadowTransform = shadowTranslate * shadowScale;

            frameData->shadow_mvp_xform_matrices[i] = shadowTransform * frameData->shadow_mvp_matrices[i];
        }
    }

    frameData->screenWidth = (float)m_view.drawableSize().width;
    frameData->fov = m_camera->fov();
}

/// Called whenever view changes orientation or layout is changed
void Renderer::drawableSizeWillChange(MTL::Size size, MTL::StorageMode GBufferStorageMode)
{
    // When reshape is called, update the aspect ratio and projection matrix since the view
    //   orientation or size has changed
    float aspect = (float)size.width / (float)size.height;
    this->m_camera->setAspect(aspect);

    MTL::TextureDescriptor GBufferTextureDesc;

    GBufferTextureDesc.pixelFormat( MTL::PixelFormatRGBA8Unorm_sRGB );
    GBufferTextureDesc.width( size.width );
    GBufferTextureDesc.height( size.height );
    GBufferTextureDesc.mipmapLevelCount( 1 );
    GBufferTextureDesc.textureType( MTL::TextureType2D );

    if(GBufferStorageMode == MTL::StorageModePrivate)
    {
        GBufferTextureDesc.usage( MTL::TextureUsageRenderTarget | MTL::TextureUsageShaderRead );
    }
    else
    {
        GBufferTextureDesc.usage( MTL::TextureUsageRenderTarget );
    }

    GBufferTextureDesc.storageMode( GBufferStorageMode );

    GBufferTextureDesc.pixelFormat( m_albedo_specular_GBufferFormat );
    m_albedo_specular_GBuffer = m_device.makeTexture( GBufferTextureDesc );

    GBufferTextureDesc.pixelFormat( m_normal_shadow_GBufferFormat );
    m_normal_shadow_GBuffer = m_device.makeTexture( GBufferTextureDesc );

    GBufferTextureDesc.pixelFormat( m_depth_GBufferFormat );
    m_depth_GBuffer = m_device.makeTexture( GBufferTextureDesc );

    m_albedo_specular_GBuffer.label( "Albedo + Spetetcular GBuffer" );
    m_normal_shadow_GBuffer.label( "Normal + Shadow GBuffer" );
    m_depth_GBuffer.label( "Depth GBuffer" );
}

#pragma mark Common Rendering Code

/// Draw the Mesh objects with the given renderEncoder
void Renderer::drawMeshes( MTL::RenderCommandEncoder & renderEncoder )
{
    for (auto& mesh : *m_meshes)
    {
        for (auto& meshBuffer : mesh.vertexBuffers())
        {
            renderEncoder.setVertexBuffer( meshBuffer.buffer(),
                                           meshBuffer.offset(),
                                           meshBuffer.argumentIndex() );
        }

        for (auto& submesh : mesh.submeshes())
        {
            // Set any textures read/sampled from the render pipeline
            const std::vector<MTL::Texture> & submeshTextures = submesh.textures();

            renderEncoder.setFragmentTexture( submeshTextures[TextureIndexBaseColor], TextureIndexBaseColor );

            renderEncoder.setFragmentTexture( submeshTextures[TextureIndexNormal], TextureIndexNormal );

            renderEncoder.setFragmentTexture( submeshTextures[TextureIndexSpecular], TextureIndexSpecular );

            renderEncoder.drawIndexedPrimitives( submesh.primitiveType(),
                                                 submesh.indexCount(),
                                                 submesh.indexType(),
                                                 submesh.indexBuffer().buffer(),
                                                 submesh.indexBuffer().offset() );
        }
    }
}

/// Get a drawable from the view (or hand back an offscreen drawable for buffer examination mode)
MTL::Texture *Renderer::currentDrawableTexture()
{
    MTL::Drawable *drawable = m_view.currentDrawable();

#if SUPPORT_BUFFER_EXAMINATION
    if(m_bufferExaminationManager->mode())
    {
        return m_bufferExaminationManager->offscreenDrawable();
    }
#endif // SUPPORT_BUFFER_EXAMINATION

    if(drawable)
    {
        return drawable->texture();
    }

    return nullptr;
}

/// Perform operations necessary at the beginning of the frame.  Wait on the in flight semaphore,
/// and get a command buffer to encode intial commands for this frame.
MTL::CommandBuffer Renderer::beginFrame()
{
    // Wait to ensure only MaxFramesInFlight are getting processed by any stage in the Metal
    // pipeline (App, Metal, Drivers, GPU, etc)

    dispatch_semaphore_wait(this->m_inFlightSemaphore, DISPATCH_TIME_FOREVER);

    // Create a new command buffer for each render pass to the current drawable
    MTL::CommandBuffer commandBuffer = m_commandQueue.commandBuffer();

    updateWorldState();

    return commandBuffer;
}

/// Perform operations necessary to obtain a command buffer for rendering to the drawable.  By
/// endoding commands that are not dependant on the drawable in a separate command buffer, Metal
/// can begin executing encoded commands for the frame (commands from the previous command buffer)
/// before a drawable for this frame becomes avaliable.
MTL::CommandBuffer Renderer::beginDrawableCommands()
{
    MTL::CommandBuffer commandBuffer = m_commandQueue.commandBuffer();

    if(!m_completedHandler)
    {
        // Create a completed handler functor for Metal to execute when the GPU has fully finished
        // processing the commands encoded for this frame.  This implenentation of the completed
        // hander signals the `m_inFlightSemaphore`, which indicates that the GPU is no longer
        // accesing the the dynamic buffer written this frame.  When the GPU no longer accesses the
        // buffer, the Renderer can safely overwrite the buffer's data to update data for a future
        // frame.
        struct CommandBufferCompletedHandler : public MTL::CommandBufferHandler
        {
            dispatch_semaphore_t semaphore;
            void operator()(const MTL::CommandBuffer &)
            {
                dispatch_semaphore_signal(semaphore);
            }
        };

        CommandBufferCompletedHandler *completedHandler = new CommandBufferCompletedHandler();
        completedHandler->semaphore = m_inFlightSemaphore;

        m_completedHandler = completedHandler;
    }

    commandBuffer.addCompletedHandler(*m_completedHandler);

    return commandBuffer;
}

/// Perform cleanup operations including presenting the drawable and committing the command buffer
/// for the current frame.  Also, when enabled, draw buffer examination elements before all this.
void Renderer::endFrame(MTL::CommandBuffer & commandBuffer)
{
#if SUPPORT_BUFFER_EXAMINATION
    if( m_bufferExaminationManager->mode() )
    {
        m_bufferExaminationManager->drawAndPresentBuffersWithCommandBuffer( commandBuffer );
    }
#endif

    // Schedule a present once the framebuffer is complete using the current drawable
    if(m_view.currentDrawable())
    {
        // Create a scheduled handler functor for Metal to present the drawable when the command
        // buffer has been scheduled by the kernel.
        struct PresentationScheduledHandler : public MTL::CommandBufferHandler
        {
            MTL::Drawable m_drawable;
            PresentationScheduledHandler(MTL::Drawable drawable)
            : m_drawable(drawable)
            {
            }

            void operator()(const MTL::CommandBuffer &)
            {
                m_drawable.present();
                delete this;
            }
        };

        PresentationScheduledHandler *scheduledHandler =
            new PresentationScheduledHandler(*m_view.currentDrawable());

        commandBuffer.addScheduledHandler(*scheduledHandler);
    }

    // Finalize rendering here & push the command buffer to the GPU
    commandBuffer.commit();
}

/// Draw to the depth texture from the directional lights point of view to generate the shadow map
void Renderer::drawShadow(MTL::CommandBuffer & commandBuffer)
{
    for (int i = 0; i < CASCADED_SHADOW_COUNT; i++) {
        m_shadowRenderPassDescriptor.depthAttachment.slice(i);

        MTL::RenderCommandEncoder encoder = commandBuffer.renderCommandEncoderWithDescriptor(m_shadowRenderPassDescriptor);

        encoder.label( "Shadow Map Pass");

        encoder.setRenderPipelineState( m_shadowGenPipelineState );
        encoder.setDepthStencilState( m_shadowDepthStencilState );
        encoder.setCullMode( MTL::CullModeBack );
        encoder.setDepthBias( 0.015, 7, 0.02 );

        encoder.setVertexBuffer( m_uniformBuffers[m_frameDataBufferIndex], 0, BufferIndexFrameData );
        encoder.setVertexBuffer(m_cascadeIndexBuffers[i], 0, BufferIndexCascadeIndex);

        drawMeshes( encoder );

        encoder.endEncoding();
    }
}

/// Draw to the three textures which compose the GBuffer
void Renderer::drawGBuffer(MTL::RenderCommandEncoder & renderEncoder)
{
    renderEncoder.pushDebugGroup( "Draw G-Buffer" );
    renderEncoder.setCullMode( MTL::CullModeBack );
    renderEncoder.setRenderPipelineState( m_GBufferPipelineState );
    renderEncoder.setDepthStencilState( m_GBufferDepthStencilState );
    renderEncoder.setStencilReferenceValue( 128 );
    renderEncoder.setVertexBuffer( m_uniformBuffers[m_frameDataBufferIndex], 0, BufferIndexFrameData );
    renderEncoder.setFragmentBuffer( m_uniformBuffers[m_frameDataBufferIndex], 0, BufferIndexFrameData );
    renderEncoder.setFragmentTexture( m_shadowMap, TextureIndexShadow );

    drawMeshes( renderEncoder );
    renderEncoder.popDebugGroup();
}

// Reduce min and max depth using compute pipeline
// It assumes that depth buffer contains depth values in eye space
void Renderer::reduceMinMaxDepth(MTL::CommandBuffer &commandBuffer)
{
    MTL::ComputeCommandEncoder computeEncoder = commandBuffer.computeCommandEncoder();

    int *dataPtrResult = (int*) m_minMaxDepthBuffer.contents();

    dataPtrResult[0] = FarPlane * LARGE_INTEGER;
    dataPtrResult[1] = NearPlane * LARGE_INTEGER;

    computeEncoder.label( "Compute min and max depth pass" );

    computeEncoder.setComputePipelineState(m_reduceComputePipelineState);
    computeEncoder.setBuffer(m_minMaxDepthBuffer, 0, BufferIndexMinMaxDepth);
    computeEncoder.setTexture(m_depth_GBuffer, TextureIndexDepth);

    MTL::Size gridSize = m_view.drawableSize();
    gridSize.depth = 1;

    unsigned long maxThreads = m_reduceComputePipelineState.maxTotalThreadsPerThreadgroup();

    MTL::Size threadgroupSize = MTL::SizeMake(sqrtl(maxThreads), sqrtl(maxThreads), 1);

    computeEncoder.dispatchThreads(gridSize, threadgroupSize);

    computeEncoder.endEncoding();
}

/// Draw the directional ("sun") light in deferred pass.  Use stencil buffer to limit execution
/// of the shader to only those pixels that should be lit
void Renderer::drawDirectionalLightCommon(MTL::RenderCommandEncoder & renderEncoder)
{
    renderEncoder.setCullMode( MTL::CullModeBack );
    renderEncoder.setStencilReferenceValue( 128 );

    renderEncoder.setRenderPipelineState( m_directionalLightPipelineState );
    renderEncoder.setDepthStencilState( m_directionLightDepthStencilState );
    renderEncoder.setVertexBuffer( m_quadVertexBuffer, 0, BufferIndexMeshPositions );
    renderEncoder.setVertexBuffer( m_uniformBuffers[m_frameDataBufferIndex], 0, BufferIndexFrameData );
    renderEncoder.setFragmentBuffer( m_uniformBuffers[m_frameDataBufferIndex], 0, BufferIndexFrameData );

    // Draw full screen quad
    renderEncoder.drawPrimitives( MTL::PrimitiveTypeTriangle, 0, 6 );
}

MTL::Library Renderer::makeShaderLibrary()
{
    CFErrorRef error = nullptr;
    CFURLRef libraryURL = nullptr;
    // macOS 11 uses shader using Metal Shading Language 2.3 which supports programmable
    // blending on Apple Silicon Macs
#ifdef TARGET_MACOS
    float osVersion = 0.0;

    {
        char osVersionString[256];
        size_t size;
        if(!sysctlbyname("kern.osrelease", osVersionString, &size, nullptr, 0))
        {
            osVersion = atof(osVersionString);
        }
    }

    if (osVersion >= 20)
    {
        libraryURL = CFBundleCopyResourceURL( CFBundleGetMainBundle() , CFSTR("MSL23Shaders"), CFSTR("metallib"), nullptr);
    }
    else
#endif
    {
        libraryURL = CFBundleCopyResourceURL( CFBundleGetMainBundle() , CFSTR("MSL20Shaders"), CFSTR("metallib"), nullptr);
    }

    MTL::Library shaderLibrary = m_device.makeLibrary(libraryURL, &error);

    AAPLAssert(!error, error, "Could not load Metal shader library");

    CFRelease(libraryURL);
    return shaderLibrary;
}

Camera* Renderer::camera()
{
    return m_camera;
}

void Renderer::changeLightPhiBy(float delta)
{
    m_lightPhi += delta;
}

void Renderer::changeLightThetaBy(float delta)
{
    m_lightTheta += delta;
}

void Renderer::switchPartitioning()
{
    if (m_partitioningMode == LOG_PARTITIONING) {
        m_partitioningMode = UNIT_PARTITIONING;
        printf("Switched to unit partitioning");
    } else {
        m_partitioningMode = LOG_PARTITIONING;
        printf("Switched to log partitioning");
    }
}
