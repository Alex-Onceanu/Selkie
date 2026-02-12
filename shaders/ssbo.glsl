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
    vec3 pos;
    int type;
    material_t material;
    vec3 scale;
    int nbNeighbours;
    int neighbours[MAX_MERGES];
    // Should be aligned to 48 bytes (12 + 4 + 16 + 12 + 4 = 48 + MAX_MERGES * 4)
};

layout(set = 0, binding = 2, std430) buffer ssbo_t {
    edit_t edits[];
} ssbo;

#endif