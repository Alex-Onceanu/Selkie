#ifndef EDITS_SSBO_H
#define EDITS_SSBO_H

// TODO : separate this in 2 ssbos : one for shape, other for material ?

struct material_t {
    vec3 albedo;
    float roughness;
};

struct edit_t {
    vec3 pos;
    int type;
    material_t material;
    vec3 scale;
    int _padding;
    // Should be aligned to 48 bytes (12 + 4 + 16 + 12 + 4 = 48)
};

layout(set = 0, binding = 2, std430) buffer e_ssbo_t {
    edit_t edits[];
} eSSBO;

#endif