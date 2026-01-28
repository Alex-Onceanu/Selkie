#ifndef SHADOW_PAYLOAD_H
#define SHADOW_PAYLOAD_H

#include "constants.glsl"

struct shadowpayload_t {
    int nbHits;
    int hitIds[MAX_MERGES];
    float softShadow;
};

#endif
