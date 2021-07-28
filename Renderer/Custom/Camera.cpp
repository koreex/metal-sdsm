//
//  Camera.cpp
//  DeferredLighting C++-macOS
//
//  Created by Koreex on 2021/7/28.
//  Copyright © 2021 Apple. All rights reserved.
//

#include <stdio.h>

#include "AAPLMathUtilities.h"
#include "Camera.h"

Camera::Camera()
{
    m_yaw = 0.5;
    m_pitch = 0.8;
    m_distance = 80;

    m_up = vector3(0.0f, 1.0f, 0.0f);
    m_center = vector3(0.0f, 5.0f, 0.0f);

    updateEye();
}

float4x4 Camera::viewMatrix()
{
    return matrix_look_at_left_hand(m_eye, m_center, m_up);
}

void Camera::updateEye()
{
    m_eye = vector3(
                    m_distance * cos(m_yaw) * sin(m_pitch) + m_center.x,
                    m_distance * cos(m_pitch) + m_center.y,
                    m_distance * sin(m_yaw) * sin(m_pitch) + m_center.z);
}

void Camera::rotateYawBy(float delta)
{
    m_yaw += delta;
    updateEye();
}

void Camera::rotatePitchBy(float delta)
{
    m_pitch += delta;
    updateEye();
}

void Camera::moveCenterBy(float deltaX, float deltaY, float deltaZ)
{

    m_center = m_center + vector3(deltaX, deltaY, deltaZ);
    updateEye();
}

void Camera::changeDistanceBy(float delta)
{
    m_distance += delta;
    updateEye();
}
