#ifndef MERGES_SSBO_H
#define MERGES_SSBO_H

struct merge_t {
    int first;
    int second;
};

layout(set = 0, binding = 3, std430) buffer m_ssbo_t {
    merge_t merges[];
} mSSBO;

#endif