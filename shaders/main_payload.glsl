#ifndef MAIN_PAYLOAD_H
#define MAIN_PAYLOAD_H

layout(push_constant) uniform PushConstants {
    float time;
};

struct payload_t {
    vec3 hitColor;
};
#endif
