#ifndef SDFS_H
#define SDFS_H

struct edit_t {
    vec3 pos;
    int type;
    float scale;
};

layout(set = 0, binding = 2, std430) buffer ssbo_t {
    edit_t edits[];
} ssbo;

// _____________________________________________________Utility________________________________________________________

mat2 rot2D(const float theta)
{
    return mat2(vec2(cos(theta), -sin(theta)), vec2(sin(theta), cos(theta)));
}

// quadratic polynomial smooth minimum
vec2 smin(const float a, const float b)
{
    const float k = 0.25; // TODO : move this in cpu

    const float h = 1.0 - min( abs(a-b)/(4.0*k), 1.0 );
    const float w = h*h;
    const float m = w*0.5;
    const float s = w*k;
    return ((a<b) ? vec2(a-s,m) : vec2(b-s,1.0-m));
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
float map(const vec3 p)
{
    vec2 ans = vec2(1. / 0., 0.5);

    for(int e = 0; e < payload.nbHits; e++)
    {
        ans = smin(ans.x, whichSdf(p, e));
    }
    return ans.x;
}
#endif
