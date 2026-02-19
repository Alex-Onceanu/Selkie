#ifndef SDFS_H
#define SDFS_H

#include "ssbo.glsl"

float norm(float n, vec2 p)
{
    return pow(pow(abs(p.x), n) + pow(abs(p.y), n), 1. / n);
}

float norm(float n, vec3 p)
{
    return pow(pow(abs(p.x), n) + pow(abs(p.y), n) + pow(abs(p.z), n), 1. / n);
}

float sdfSphere(const float n, const vec3 p, const float radius)
{
    return norm(n, p) - radius;
}

float sdfBox(const float n, const vec3 p, const vec3 b)
{
    vec3 q = abs(p) - b;
    return norm(n, max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

float sdfTorus(const float n, const vec3 p, const vec2 t)
{
    vec2 q = vec2(norm(n, p.xz)-t.x,p.y);
    return norm(n, q)-t.y;
}

float sdfCapsule(const float n, vec3 p, const vec2 s)
{
    p.y -= clamp(p.y, 0.0, s.x);
    return norm(n, p) - s.y;
}

float sdfCylinder(const float n, const vec3 p, const vec2 s)
{
    vec2 d = abs(vec2(norm(n, p.xz),p.y)) - s;
    return min(max(d.x,d.y),0.0) + norm(n, max(d,0.0));
}

float sdfRoundCone(const float n, const vec3 p, const vec3 s)
{
    const float r1 = s.x;
    const float r2 = s.y;
    const float h = s.z;
    float b = (r1-r2)/h;
    float a = sqrt(1.0-b*b);

    vec2 q = vec2( norm(n, p.xz), p.y );
    float k = dot(q,vec2(-b,a));
    if( k<0.0 ) return norm(n, q) - r1;
    if( k>a*h ) return norm(n, q-vec2(0.0,h)) - r2;
    return dot(q, vec2(a,b) ) - r1;
}

vec3 opBend(vec3 p, float k)
{
    float c = cos(k*p.x);
    float s = sin(k*p.x);
    mat2  m = mat2(c,-s,s,c);
    return vec3(m*p.xy,p.z);
}

vec3 opTwist(vec3 p, float k)
{
    if(k <= 0.) return p;
    float c = cos(k*p.y);
    float s = sin(k*p.y);
    mat2  m = mat2(c,-s,s,c);
    return vec3(m*p.xz,p.y);
}

// returns signed distance from p to the object
float sdf(const vec3 p, const int which)
{
    const edit_t e = ssbo.edits[which];
    const mat4 model = mat4(e.transform_l1.x, e.transform_l1.y, e.transform_l1.z, e.transform_l1.w,
                            e.transform_l2.x, e.transform_l2.y, e.transform_l2.z, e.transform_l2.w,
                            e.transform_l3.x, e.transform_l3.y, e.transform_l3.z, e.transform_l3.w,
                            0., 0., 0., 1.);

    vec3 rp = (vec4(p, 1.) * model).xyz;
    rp = opBend(opTwist(rp, e.twist), e.bend);
    rp /= e.scale;
    vec3 elongation_rp = abs(rp) - e.elongation;
    if(length(e.elongation) > 0.) rp = max(vec3(0.), elongation_rp);

    float d = 1. / 0.;
    const float n = e.norm;
    switch(e.type)
    {
    case 0:
        d = sdfSphere(n, rp, e.dimensions.x);
        break;
    case 1:
        d = sdfBox(n, rp, e.dimensions);
        break;
    case 2:
        d = sdfTorus(n, rp, e.dimensions.xy);
        break;
    case 3:
        d = sdfCapsule(n, rp, e.dimensions.xy);
        break;
    case 4:
        d = sdfCylinder(n, rp, e.dimensions.xy);
        break;
    case 5:
        d = sdfRoundCone(n, rp, e.dimensions);
        break;
    default:
        break;
    }

    if(length(e.elongation) > 0.) d += min(max(elongation_rp.x,max(elongation_rp.y,elongation_rp.z)), 0.0);
    d *= e.scale;
    d -= e.rounding;

    return d;
}

#endif
