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

#endif /* SDSM_Utilities_h */
