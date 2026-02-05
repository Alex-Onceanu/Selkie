#ifndef SDFS_H
#define SDFS_H

// TODO : material_t here instead of clr and roughness
struct edit_t {
    vec3 pos;
    int type;
    vec3 clr;
    float roughness;
    float scale;
    float padding[7];
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

float sdfSphere(const vec3 p, const vec3 center, const float radius)
{
    return length(p - center) - radius;
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
    switch(ssbo.edits[payload.hitIds[which]].type)
    {
    case 0:
        return sdfSphere(p, ssbo.edits[payload.hitIds[which]].pos, ssbo.edits[payload.hitIds[which]].scale);
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
