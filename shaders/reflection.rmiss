#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_debug_printf : enable

#include "main_payload.glsl"
#include "shadow_payload.glsl"

layout(location = 1) rayPayloadInEXT shadowpayload_t shadowPayload;
layout(location = 2) rayPayloadInEXT payload_t payload;

layout(set = 0, binding = 0) uniform accelerationStructureEXT bvh;

vec3 mirrorRay(const vec3 ro, const vec3 rd)
{
    return vec3(1., 1., 1.);
}

#include "raymarch.glsl"

void main()
{
    raymarch(REFLECT_MAX_IT);
}