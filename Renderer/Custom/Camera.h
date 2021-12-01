//
//  Camera.h
//  DeferredLighting C++-macOS
//
//  Created by Koreex on 2021/7/28.
//  Copyright © 2021 Apple. All rights reserved.
//

#ifndef Camera_h
#define Camera_h

#include <simd/simd.h>

using namespace simd;

class Camera
{
public:

    explicit Camera();

    float4x4 viewMatrix();
    float4x4 projMatrix();
    void rotateYawBy(float delta);
    void rotatePitchBy(float delta);
    void moveCenterBy(float deltaX, float deltaY, float deltaZ);
    void changeDistanceBy(float delta);
    void setFov(float fov);
    void setAspect(float aspect);
    void setNear(float near);
    void setFar(float far);
    float aspect();
    float fov();
    float3 position();

private:

    vector_float3 m_eye;
    vector_float3 m_up;
    vector_float3 m_center;
    vector_float3 m_forward;
    vector_float3 m_tangent;
    float m_distance;
    float m_pitch;
    float m_yaw;

    float m_fov;
    float m_aspect;
    float m_near;
    float m_far;

    void updateEye();
};

#endif /* Camera_h */
