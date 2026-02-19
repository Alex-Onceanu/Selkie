#ifndef SSBO_H
#define SSBO_H

// TODO : separate this in 2 ssbos : one for shape, other for material ?
// Should be a multiple of 4 for gpu memory alignment !!!
#define MAX_MERGES 16

struct material_t {
    vec3 albedo;
    float roughness;
};

struct edit_t {
    vec4        transform_l1,
                transform_l2, 
                transform_l3;
    material_t  material;
    vec3        dimensions;
    int         type;
    vec3        elongation;
    float       rounding;
    float       scale;
    bool        glass;
    float       blendStrength;
    bool        negative;
    float       bend;
    float       norm;
    float       twist;
    int         nbNeighbours;
    int         neighbours[MAX_MERGES];
};

layout(set = 0, binding = 2, std430) buffer ssbo_t {
    edit_t edits[];
} ssbo;

vec3 getPos(int i)
{
    edit_t e = ssbo.edits[i];
    return vec3(e.transform_l1.w, e.transform_l2.w, e.transform_l3.w);
}

#endif