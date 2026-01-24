#pragma once

#define CLAMP(val, minval, maxval) (val > maxval ? maxval : (val < minval ? minval : val))

namespace math
{
    struct vec2
    {
        float x, y;

        vec2(float __x, float __y) { x = __x; y = __y; };
    };

    struct vec3
    {
        float x, y, z;

        vec3(float __x, float __y, float __z) { x = __x; y = __y; z = __z; };
    };

    struct vec4
    {
        float x, y, z, w;

        vec4(float __x, float __y, float __z, float __w) { x = __x; y = __y; z = __z; w = __w; };
    };
}