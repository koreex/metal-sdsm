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

void uniformPartitioning(float min, float max, int partitionCount, float *result)
{
    for (uint i = 0; i < partitionCount + 1; i++) {
        result[i] = min + (max - min) * (float)i / (float)partitionCount;
    }
}


float4x4 cascadedShadowProjectionMatrix(float4x4 cameraViewMatrix, float aspectRatio, float fov,
                                float4x4 shadowViewMatrix,
                                float *cascadeEnds, int index,
                                FrustumVertex *viewFrustumBuffer, FrustumVertex *lightFrustumBuffer,
                                MTL::Buffer lightFrustumBoundingBox)
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
        vector4(-xn, -yn, cascadeEnds[index], 1.0f),
        vector4(xn, -yn, cascadeEnds[index], 1.0f),

        vector4(xf, yf, cascadeEnds[index + 1], 1.0f),
        vector4(-xf, yf, cascadeEnds[index + 1], 1.0f),
        vector4(-xf, -yf, cascadeEnds[index + 1], 1.0f),
        vector4(xf, -yf, cascadeEnds[index + 1], 1.0f),
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

        if (j < 4) {
            viewFrustumBuffer[index * 4 + j] = {{vW.x, vW.y, vW.z}, {1.0f, 1.0f, 1.0f}};
        }

        if (index == CASCADED_SHADOW_COUNT - 1 && j >= 4) {
            viewFrustumBuffer[index * 4 + j] = {{vW.x, vW.y, vW.z}, {1.0f, 1.0f, 1.0f}};
        }
    }

    int *dataPtr = (int*) lightFrustumBoundingBox.contents();

    minX = (float)dataPtr[6 * index + BoundingBoxMinX] / (float)LARGE_INTEGER;
    minY = (float)dataPtr[6 * index + BoundingBoxMinY] / (float)LARGE_INTEGER;
    minZ = (float)dataPtr[6 * index + BoundingBoxMinZ] / (float)LARGE_INTEGER;
    maxX = (float)dataPtr[6 * index + BoundingBoxMaxX] / (float)LARGE_INTEGER;
    maxY = (float)dataPtr[6 * index + BoundingBoxMaxY] / (float)LARGE_INTEGER;
    maxZ = (float)dataPtr[6 * index + BoundingBoxMaxZ] / (float)LARGE_INTEGER;

    float4 lightFrustumCornersSV[8];

    lightFrustumCornersSV[0] = {minX, minY, minZ, 1.0f};
    lightFrustumCornersSV[1] = {maxX, minY, minZ, 1.0f};
    lightFrustumCornersSV[2] = {maxX, maxY, minZ, 1.0f};
    lightFrustumCornersSV[3] = {minX, maxY, minZ, 1.0f};
    lightFrustumCornersSV[4] = {minX, minY, maxZ, 1.0f};
    lightFrustumCornersSV[5] = {maxX, minY, maxZ, 1.0f};
    lightFrustumCornersSV[6] = {maxX, maxY, maxZ, 1.0f};
    lightFrustumCornersSV[7] = {minX, maxY, maxZ, 1.0f};

    for (uint i = 0; i < 8; i++) {
        float4 corner = matrix_invert(shadowViewMatrix) * lightFrustumCornersSV[i];
        lightFrustumBuffer[index * 8 + i] = {{corner.x, corner.y, corner.z}, {0.0f, 1.0f, 0.0f}};
    }

    return matrix_ortho_left_hand(minX, maxX, minY, maxY, -100, maxZ);
}

#endif /* SDSM_Utilities_h */
