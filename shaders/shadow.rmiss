#version 460
#extension GL_EXT_ray_tracing : require

#extension GL_GOOGLE_include_directive : require

#include "shadow_payload.glsl"

layout(location = 1) rayPayloadInEXT shadowpayload_t payload;

layout(set = 0, binding = 0) uniform accelerationStructureEXT bvh;

#include "sdfs.glsl"

#define SOFTNESS 16.
void main()
{
    if(payload.nbHits <= 0) return;

    vec3 p = gl_WorldRayOriginEXT;
    const vec3 rd = normalize(gl_WorldRayDirectionEXT);

    float t = 0.1; p += t * rd;  // do not change this !
    for(int i = 0; i < SHADOW_MAX_IT; i++)
    {
        float safeDist = map(p);

        payload.softShadow = min(payload.softShadow, SOFTNESS * safeDist / t);

        p += rd * safeDist;
        t += safeDist;

        if(t > gl_RayTmaxEXT || safeDist <= gl_RayTminEXT) break;
    }
}