#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_debug_printf : enable

#include "util.glsl"
#include "sdfs.glsl"

hitAttributeEXT hitInfo_t hitInfo;

layout(push_constant) uniform PushConstants {
    float time;
};

layout(set = 0, binding = 2, std430) buffer e_ssbo_t {
    edit_t edits[];
} eSSBO;

layout(location = 0) rayPayloadInEXT payload_t payload;
layout(location = 1) rayPayloadEXT shadowPayload_t shadowPayload;


layout(set = 0, binding = 0) uniform accelerationStructureEXT bvh;

/* TODO : change sbtRecordOffset 
        void traceRayEXT(accelerationStructureEXT topLevel,
                   uint rayFlags,
                   uint cullMask,
                   uint sbtRecordOffset,
                   uint sbtRecordStride,
                   uint missIndex,
                   vec3 origin,
                   float Tmin,
                   vec3 direction,
                   float Tmax,
                   int payload);
*/
float shadowRay(const vec3 ro, const vec3 rd)
{
    // return 1.;
    // removing the shadow makes the weird 1s lag spike disappear...
    shadowPayload.shadow = 1.;
    traceRayEXT(bvh, gl_RayFlagsSkipClosestHitShaderEXT, 0xFF, 0, 0, 1, ro, T_MIN, rd, T_MAX, 1);

    return clamp(shadowPayload.shadow, AMBIENT_INTENSITY, 1.);
}

vec3 mirrorRay(const vec3 ro, const vec3 rd)
{
    return vec3(1.);
    // mirrorPayload.nbHits = 0;
    // traceRayEXT(bvh, gl_RayFlagsSkipClosestHitShaderEXT, 0xFF, 0, 0, 0, ro + 0.01 * rd, T_MIN, rd, T_MAX, 0);
    
    // return mirrorPayload.hitColor;
}

vec3 sphereColor(const vec3 p, const vec3 rd, const vec3 normal, const material_t mat, const vec3 lightPos)
{
    const vec3 toLight = normalize(lightPos - p);
    const float diffuse = max(AMBIENT_INTENSITY, dot(normal, toLight));

    const float shadow = shadowRay(p, toLight);
    const vec3 mir = mirrorRay(p, reflect(rd, normal));

    return mix(mir, mat.albedo, mat.roughness) * min(shadow, diffuse);
}

void main()
{
    const vec3 p = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT / length(gl_WorldRayDirectionEXT);
    vec3 lp = LIGHTPOS;
    lp.xz *= rot2D(-2.7 * time);
    payload.hitColor = sphereColor(p, gl_WorldRayDirectionEXT, hitInfo.normal, eSSBO.edits[gl_PrimitiveID].material, lp);
}