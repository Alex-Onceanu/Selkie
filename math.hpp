#pragma once

#include <cmath> // sqrtf

#define CLAMP(val, minval, maxval) (val > maxval ? maxval : (val < minval ? minval : val))

namespace sk::math
{
    struct vec2
    {
        float x = 0.f, y = 0.f;

        vec2() = default;
        vec2(float __x, float __y) { x = __x; y = __y; };

        vec2& operator+=(const vec2 &r) { x += r.x; y += r.y; return *this; }
        vec2& operator-=(const vec2 &r) { x -= r.x; y -= r.y; return *this; }
        vec2& operator*=(const float s) { x *= s; y *=s; return *this; }
        vec2 operator+(const vec2 &r) const { return vec2(*this) += r; }
        vec2 operator-(const vec2 &r) const { return vec2(*this) -= r; }

        vec2 operator*(const float r) const { return vec2(*this) *= r; }
        vec2 operator-() const { return vec2(-x, -y); }

        float dot(const vec2& r) const { return x * r.x + y * r.y; }
        float length() const { return sqrtf(dot(*this)); }
        vec2 normalize() const { return *this * (1. / length()); }
    };

    struct vec3
    {
        float x = 0.f, y = 0.f, z = 0.f;

        vec3(float __x, float __y, float __z) { x = __x; y = __y; z = __z; };
        vec3(float __xyz) { x = __xyz; y = __xyz; z = __xyz; };
        vec3() = default;

        vec3& operator+=(const vec3 &r) { x += r.x; y += r.y; z += r.z; return *this; }
        vec3& operator-=(const vec3 &r) { x -= r.x; y -= r.y; z -= r.z; return *this; }
        vec3& operator*=(const float s) { x *= s; y *=s ; z *= s; return *this; }
        vec3 operator+(const vec3 &r) const { return vec3(*this) += r; }
        vec3 operator-(const vec3 &r) const { return vec3(*this) -= r; }

        vec3 operator*(const float r) const { return vec3(*this) *= r; }
        vec3 operator-() const { return vec3(-x, -y, -z); }

        float dot(const vec3& r) const { return x * r.x + y * r.y + z * r.z; }
        vec3 cross(const vec3& r) const { return vec3(y*r.z - z*r.y, z*r.x - x*r.z, x*r.y - y*r.x); }
        float length() const { return sqrtf(dot(*this)); }
        vec3 normalize() const { return *this * (1. / length()); }
    };

    struct vec4
    {
        float x = 0.f, y = 0.f, z = 0.f, w = 0.f;

        vec4() = default;
        vec4(const float x__, const float y__, const float z__, const float w__) { x = x__, y = y__, z = z__, w = w__; };
        vec4(const vec3 xyz__, const float w__) { x = xyz__.x, y = xyz__.y, z = xyz__.z, w = w__; }

        vec3 xyz() const { return math::vec3(x, y, z); }
    };

    struct mat3
    {
        vec3 C1, C2, C3;

        mat3() : C1(vec3(1.,0.,0.)), C2(vec3(0.,1.,0.)), C3(vec3(0.,0.,1.)) {}
        mat3(vec3 m[3]) { C1 = m[0]; C2 = m[1]; C3 = m[2]; }
        mat3(const vec3 c1, const vec3 c2, const vec3 c3) : C1(c1), C2(c2), C3(c3) {}
        mat3(const float m11, const float m12, const float m13, 
             const float m21, const float m22, const float m23, 
             const float m31, const float m32, const float m33)
             : mat3(vec3(m11, m21, m31), vec3(m12, m22, m32), vec3(m13, m23, m33)) {}

        mat3 transpose() const { return mat3(vec3(C1.x, C2.x, C3.x), vec3(C1.y, C2.y, C3.y), vec3(C1.z, C2.z, C3.z)); }
        vec3 operator*(const vec3& v) const { return vec3(v.dot(vec3(C1.x, C2.x, C3.x)), v.dot(vec3(C1.y, C2.y, C3.y)), v.dot(vec3(C1.z, C2.z, C3.z))); }
        mat3 operator*(const mat3& o) const
        {
            mat3 T = transpose();
            return mat3(vec3(T.C1.dot(o.C1), T.C1.dot(o.C2), T.C1.dot(o.C3)), 
                        vec3(T.C2.dot(o.C1), T.C2.dot(o.C2), T.C2.dot(o.C3)), 
                        vec3(T.C3.dot(o.C1), T.C3.dot(o.C2), T.C3.dot(o.C3)));
        }

        void coefs(float* p) const { if(!p) return;
            p[0] = C1.x;    p[3] = C2.x;   p[6] = C3.x; 
            p[1] = C1.y;    p[4] = C2.y;   p[7] = C3.y;
            p[2] = C1.z;    p[5] = C2.z;   p[8] = C3.z;
        }
    };

    struct Quaternion
    {
        float s{};
        vec3 w{};

        Quaternion(const float s_, const vec3& w_) : s(s_), w(w_) {}
        Quaternion(const vec3 axis, const float angle)
        {
            s = cosf(angle / 2.f);
            w = axis * sinf(angle / 2.f);
        }

        Quaternion operator+(const Quaternion& o) { return Quaternion(s + o.s, w + o.w); }
        Quaternion operator-(const Quaternion& o) { return Quaternion(s - o.s, w - o.w); }
        Quaternion operator*(const Quaternion& o)
        {
            return Quaternion(s * o.s - w.dot(o.w), o.w * s + w * o.s - w.cross(o.w));
        }
        Quaternion operator*(const float f) { return Quaternion(s * f, w * f); }

        Quaternion& operator+=(const Quaternion& o) { return *this = (*this + o); }
        Quaternion& operator-=(const Quaternion& o) { return *this = (*this - o); }
        Quaternion& operator*=(const Quaternion& o) { return *this = (*this * o); }
        Quaternion& operator*=(const float f) { return *this = (*this * f); }

        mat3 toMatrix()
        {
            return mat3(1.-2.*w.y*w.y-2.*w.z*w.z, 2.*w.x*w.y+2.*s*w.z, 2.*w.x*w.z-2.*s*w.y,
                        2.*w.x*w.y-2.*s*w.z, 1.-2.*w.x*w.x-2.*w.z*w.z, 2.*w.y*w.z+2.*s*w.x,
                        2.*w.x*w.z+2.*s*w.y, 2.*w.y*w.z-2.*s*w.x, 1.-2.*w.x*w.x-2.*w.y*w.y);
        }

        // inplace (builder pattern)
        Quaternion& normalized() { return *this *= (1.f / sqrtf(s * s + w.dot(w))); }
    };
}