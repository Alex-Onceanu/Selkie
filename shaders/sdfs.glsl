#ifndef SDFS_H
#define SDFS_H

float sdfSphere(const vec3 p, const float radius)
{
    return length(p) - radius;
}

float sdfBox(const vec3 p, const vec3 b)
{
    float r = 0.0; // rounded box
    vec3 q = abs(p) - b + r;
    return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0) - r;
}

float sdfTorus(const vec3 p, const vec2 t)
{
    vec2 q = vec2(length(p.xz)-t.x,p.y);
    return length(q)-t.y;
}

float sdfCapsule(vec3 p, const vec2 s)
{
    p.y -= clamp(p.y, 0.0, s.x);
    return length(p) - s.y;
}

float sdfCylinder(const vec3 p, const vec2 s)
{
    vec2 d = abs(vec2(length(p.xz),p.y)) - s;
    return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

float sdfRoundCone(const vec3 p, const vec3 s)
{
    const float r1 = s.x;
    const float r2 = s.y;
    const float h = s.z;
    float b = (r1-r2)/h;
    float a = sqrt(1.0-b*b);

    vec2 q = vec2( length(p.xz), p.y );
    float k = dot(q,vec2(-b,a));
    if( k<0.0 ) return length(q) - r1;
    if( k>a*h ) return length(q-vec2(0.0,h)) - r2;
    return dot(q, vec2(a,b) ) - r1;
}

// returns signed distance from p to the object (type, objPos, objScale) 
// float sdf(const vec3 p, const int type, const vec3 objPos, const vec3 objScale)
// {
//     vec3 relPos = objPos - p;
//     switch(type)
//     {
//     case 0:
//         return sdfSphere(relPos, objScale.x);
//     case 1:
//         return sdfBox(relPos, objScale);
//     case 2:
//         return sdfTorus(relPos, objScale.xy);
//     case 3:
//         return sdfCapsule(relPos, objScale.xy);
//     case 4:
//         return sdfCylinder(relPos, objScale.xy);
//     case 5:
//         return sdfRoundCone(relPos, objScale);
//     default:
//         break;
//     }
//     return 1. / 0.;
// }


// returns signed distance from p to the object
float sdf(const vec3 p, const int type, const vec3 dims, const mat4 model)
{
    vec3 rp = (model * vec4(p, 1.)).xyz;
    switch(type)
    {
    case 0:
        return sdfSphere(rp, dims.x);
    case 1:
        return sdfBox(rp, dims);
    case 2:
        return sdfTorus(rp, dims.xy);
    case 3:
        return sdfCapsule(rp, dims.xy);
    case 4:
        return sdfCylinder(rp, dims.xy);
    case 5:
        return sdfRoundCone(rp, dims);
    default:
        break;
    }
    return 1. / 0.;
}


#endif
