//
//  SDSM_Utilities.h
//  DeferredLighting C++
//
//  Created by Koreex on 2021/11/8.
//  Copyright © 2021 Apple. All rights reserved.
//

#ifndef SDSM_Utilities_h
#define SDSM_Utilities_h

#include "AAPLConfig.h"

void logPartitioning(float min, float max, int partitionCount, float *result)
{
    for (uint i = 0; i < partitionCount + 1; i++) {
        result[i] = pow(max / min, (float)i / (float)partitionCount) * min;
    }
}

float4x4 cascadedShadowProjectionMatrix(float4x4 cameraViewMatrix, float aspectRatio, float fov,
                                float4x4 shadowViewMatrix,
                                float *cascadeEnds, int index)
{
    float tanHalfHFov = tanf(fov / 2) * aspectRatio;
    float tanHalfVFov = tanf(fov / 2);

    float xn = cascadeEnds[index] * tanHalfHFov;
    float xf = cascadeEnds[index + 1] * tanHalfHFov;
    float yn = cascadeEnds[index] * tanHalfVFov;
    float yf = cascadeEnds[index + 1] * tanHalfVFov;

    float4 frustumCorners[8] = {
        vector4(xn, yn, cascadeEnds[index], 1.0f),
        vector4(-xn, yn, cascadeEnds[index], 1.0f),
        vector4(xn, -yn, cascadeEnds[index], 1.0f),
        vector4(-xn, -yn, cascadeEnds[index], 1.0f),

        vector4(xf, yf, cascadeEnds[index + 1], 1.0f),
        vector4(-xf, yf, cascadeEnds[index + 1], 1.0f),
        vector4(xf, -yf, cascadeEnds[index + 1], 1.0f),
        vector4(-xf, -yf, cascadeEnds[index + 1], 1.0f),
    };

    float4 frustumCornersL[8];

    float numericLimit = 1.0e10;

    float minX = numericLimit;
    float maxX = -numericLimit;
    float minY = numericLimit;
    float maxY = -numericLimit;
    float minZ = numericLimit;
    float maxZ = -numericLimit;

    for (uint j = 0; j < 8; j++) {
        float4 vW = matrix_invert(cameraViewMatrix) * frustumCorners[j];
        frustumCornersL[j] = shadowViewMatrix * vW;

        minX = min(minX, frustumCornersL[j].x);
        maxX = max(maxX, frustumCornersL[j].x);
        minY = min(minY, frustumCornersL[j].y);
        maxY = max(maxY, frustumCornersL[j].y);
        minZ = min(minZ, frustumCornersL[j].z);
        maxZ = max(maxZ, frustumCornersL[j].z);
    }

    return matrix_ortho_left_hand(minX, maxX, minY, maxY, -100, maxZ);
}

#endif /* SDSM_Utilities_h */
