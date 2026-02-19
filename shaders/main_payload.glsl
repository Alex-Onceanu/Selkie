#ifndef MAIN_PAYLOAD_H
#define MAIN_PAYLOAD_H

layout(push_constant) uniform PushConstants {
    float time;
};

struct payload_t {
    vec3 hitColor;
    float shadow;
    
    bool mirrorRay;
    vec3 mir_ro;
    vec3 mir_rd;
    float mir_rough;

    bool refractRay;
    vec3 refr_ro;
    vec3 refr_rd;
    float negativeRay;
};

mat2 rot2D(const float theta)
{
    return mat2(vec2(cos(theta), -sin(theta)), vec2(sin(theta), cos(theta)));
}

#endif
