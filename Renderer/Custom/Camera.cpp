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

    m_forward = normalize(m_center - m_eye);
    m_tangent = normalize(cross(m_forward, vector3(0.0f, 1.0f, 0.0f)));
    m_up = cross(m_tangent, m_forward);

    m_fov = 65.0f * (M_PI / 180.0f);
}

void Camera::rotateYawBy(float delta)
{
    m_yaw += delta;
    updateEye();
}

void Camera::rotatePitchBy(float delta)
{
    m_pitch += delta;

    if (m_pitch > M_PI) {
        m_pitch = M_PI;
    }

    if (m_pitch < 0) {
        m_pitch = 1e-5;
    }
    updateEye();
}

void Camera::moveCenterBy(float deltaX, float deltaY, float deltaZ)
{

    m_center = m_center + deltaX * m_tangent + deltaY * m_up;
    updateEye();
}

void Camera::changeDistanceBy(float delta)
{
    m_distance += delta;
    updateEye();
}

void Camera::setFov(float fov)
{
    m_fov = fov;
}

void Camera::setAspect(float aspect)
{
    m_aspect = aspect;
}

void Camera::setNear(float near)
{
    m_near = near;
}

void Camera::setFar(float far)
{
    m_far = far;
}

float4x4 Camera::projMatrix()
{
    return matrix_perspective_left_hand(m_fov, m_aspect, m_near, m_far);
}

float Camera::aspect()
{
    return m_aspect;
}

float Camera::fov()
{
    return m_fov;
}

float3 Camera::position()
{
    return m_eye;
}
