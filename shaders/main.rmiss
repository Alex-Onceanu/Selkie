#version 460
#extension GL_EXT_ray_tracing : require

#include "main_payload.glsl"
#include "shadow_payload.glsl"

layout(location = 0) rayPayloadInEXT payload_t payload;
layout(location = 1) rayPayloadInEXT shadowpayload_t shadowPayload;
layout(location = 2) rayPayloadInEXT payload_t mirrorPayload;

layout(set = 0, binding = 0) uniform accelerationStructureEXT bvh;

vec3 mirrorRay(const vec3 ro, const vec3 rd)
{
    mirrorPayload.nbHits = 0;
    traceRayEXT(bvh, gl_RayFlagsSkipClosestHitShaderEXT, 0xFF, 0, 0, 2, ro + 0.1 * rd, T_MIN, rd, T_MAX, 2);
    
    return mirrorPayload.hitColor;
}

#include "raymarch.glsl"

void main()
{
    raymarch(MAX_IT);
}