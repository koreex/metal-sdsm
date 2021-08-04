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
    void rotateYawBy(float delta);
    void rotatePitchBy(float delta);
    void moveCenterBy(float deltaX, float deltaY, float deltaZ);
    void changeDistanceBy(float delta);

private:

    vector_float3 m_eye;
    vector_float3 m_up;
    vector_float3 m_center;
    vector_float3 m_forward;
    vector_float3 m_tangent;
    float m_distance;
    float m_pitch;
    float m_yaw;

    void updateEye();
};

#endif /* Camera_h */
