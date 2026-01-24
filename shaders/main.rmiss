#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_debug_printf : enable

struct Edit {
    vec4 pos;
};

layout(set = 0, binding = 2, std430) buffer ssbo_t {
    Edit edits[];
} ssbo;

#define MAX_MERGES 8
struct payload_t {
    float time;
    vec3 hitColor;
    float minDist;
    float distToMinDist;
    float tHit;
    int nbHits;
    int hitIds[MAX_MERGES];
};

layout(location = 0) rayPayloadInEXT payload_t payload;

void main()
{
    // debugPrintfEXT("Edit pos of 0 : %f, %f, %f, %f", ssbo.edits[0].pos.x, ssbo.edits[0].pos.y, ssbo.edits[0].pos.z, ssbo.edits[0].pos.w);
    if(payload.nbHits > 1) {
        if(ssbo.edits[0].pos.x > 0.5) payload.hitColor = vec3(0.2, 0.2, 0.2);
        else payload.hitColor = vec3(0.0, 0.0, 1.0);
    } else if(payload.nbHits > 0)
        payload.hitColor = vec3(1., 0., 1.0);
    else
        payload.hitColor = vec3(1.0, 0.0, 0.0);
}