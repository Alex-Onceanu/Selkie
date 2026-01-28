#ifndef MAIN_PAYLOAD_H
#define MAIN_PAYLOAD_H

#include "constants.glsl"

struct payload_t {
    int nbHits;
    int hitIds[MAX_MERGES];
    vec3 hitColor;
};
#endif
