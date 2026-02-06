#ifndef SDFS_H
#define SDFS_H

struct material_t {
    vec3 albedo;
    float roughness;
};

struct edit_t {
    vec3 pos;
    int type;
    material_t material;
    vec3 scale;
    float padding[4]; // Align to 48 bytes (12 + 4 + 16 + 12 + padding 4 = 48)
};

layout(set = 0, binding = 2, std430) buffer ssbo_t {
    edit_t edits[];
} ssbo;

// _____________________________________________________Utility________________________________________________________

mat2 rot2D(const float theta)
{
    return mat2(vec2(cos(theta), -sin(theta)), vec2(sin(theta), cos(theta)));
}

// returns a negative value if there is no intersection
float rayPlane(const vec3 ro, const vec3 rd)
{
    return -ro.y / rd.y;
}

// ________________________________________________________________________SDFs__________________________________________________________________

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

float sdfCapsule(vec3 p, const float h, const float r)
{
    p.y -= clamp( p.y, 0.0, h );
    return length( p ) - r;
}

float sdfCylinder(const vec3 p, const float r, const float h)
{
    vec2 d = abs(vec2(length(p.xz),p.y)) - vec2(r,h);
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

// _______________________________________________________________________Scene__________________________________________________________________

// if there is an intersection with the infinite ground plane before t, updates p (inplace) and returns true
bool intersectsGround(const vec3 ro, const vec3 rd, const float t, out vec3 p)
{
    float tGround = rayPlane(ro, rd);
    if(tGround > gl_RayTminEXT && tGround < t)
    {
        p = ro + tGround * rd;
        return true;
    }
    return false;
}

// returns signed distance from p to the closest edit_t in edits[payload.hitIds[...]] 
float whichSdf(const vec3 p, const int which)
{
    const vec3 obj_p = p - ssbo.edits[payload.hitIds[which]].pos;
    switch(ssbo.edits[payload.hitIds[which]].type)
    {
    case 0:
        return sdfSphere(obj_p, ssbo.edits[payload.hitIds[which]].scale.x);
    case 1:
        return sdfBox(obj_p, vec3(ssbo.edits[payload.hitIds[which]].scale));
    case 2:
        return sdfTorus(obj_p, vec2(ssbo.edits[payload.hitIds[which]].scale.xy));
    case 3:
        return sdfCapsule(obj_p, ssbo.edits[payload.hitIds[which]].scale.x, ssbo.edits[payload.hitIds[which]].scale.y);
    case 4:
        return sdfCylinder(obj_p, ssbo.edits[payload.hitIds[which]].scale.x, ssbo.edits[payload.hitIds[which]].scale.y);
    case 5:
        return sdfRoundCone(obj_p, ssbo.edits[payload.hitIds[which]].scale);
    default:
        break;
    }
    return 1. / 0.;
}

// returns distance to the closest object in the scene
// computes n-ary exponential smooth minimum 
float map(const vec3 p)
{
    float sum = 0.;
    for(int e = 0; e < payload.nbHits; e++)
    {
        sum += exp2(-whichSdf(p, e) * BLEND_STRENGTH);
    }
    return -log2(sum) / BLEND_STRENGTH;
}
#endif
