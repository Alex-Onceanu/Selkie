#pragma once

#include <cmath> // sqrtf

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
        vec3() = default;

        vec3& operator+=(const vec3 &r) { x += r.x; y += r.y; z += r.z; return *this; }
        vec3& operator-=(const vec3 &r) { x -= r.x; y -= r.y; z -= r.z; return *this; }
        vec3& operator*=(const float s) { x *= s; y *=s ; z *= s; return *this; }
        vec3 operator+(const vec3 &r) const { return vec3(*this) += r; }
        vec3 operator-(const vec3 &r) const { return vec3(*this) -= r; }

        vec3 operator*(const float r) const { return vec3(*this) *= r; }
        vec3 operator-() const { return vec3(-x, -y, -z); }


        float dot(const vec3& r) { return x * r.x + y * r.y + z * r.z; }
        float length() { return sqrtf(dot(*this)); }
    };

    struct vec4
    {
        float x, y, z, w;

        vec4(float __x, float __y, float __z, float __w) { x = __x; y = __y; z = __z; w = __w; };
    };
}