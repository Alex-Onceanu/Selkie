#version 460
#extension GL_EXT_ray_tracing : require

struct Edit {
    vec4 pos;
    int type;
    float scale;
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
    if(payload.nbHits > 1) {
        if(ssbo.edits[0].type == 1) payload.hitColor = vec3(1.0, 1.0, 0.0);
        else                        payload.hitColor = vec3(0.0, 1.0, 0.0);
    } else if(payload.nbHits > 0)
        payload.hitColor = vec3(0., 0., 1.0);
    else
        payload.hitColor = vec3(0.5, 0.3, 0.7);
}