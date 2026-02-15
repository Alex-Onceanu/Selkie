#ifndef SSBO_H
#define SSBO_H

// TODO : separate this in 2 ssbos : one for shape, other for material ?
// Should be a multiple of 4 for gpu memory alignment !!!
#define MAX_MERGES 8

struct material_t {
    vec3 albedo;
    float roughness;
};

struct edit_t {
    vec4        transform_l1,
                transform_l2, 
                transform_l3;
    material_t  material;
    int         type;
    vec3        dimensions;
    float       rounding;
    vec3        elongation;
    float       blendStrength;
    vec3        scale;
    bool        negative;
    vec3        bend;
    float       onion;
    float       norm;
    float       twist;
    int         nbNeighbours;
    int         neighbours[MAX_MERGES];
    // Should be aligned to 48 bytes (12 + 4 + 16 + 12 + 4 = 48 + MAX_MERGES * 4)
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