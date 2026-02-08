#ifndef PBR_H
#define PBR_H

#include "constants.glsl"

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
    return 1.;
    // shadowPayload.nbHits = 0;
    // shadowPayload.softShadow = 1.;
    // traceRayEXT(bvh, gl_RayFlagsSkipClosestHitShaderEXT, 0xFF, 0, 0, 0, ro, T_MIN, rd, T_MAX, 0);

    // return clamp(shadowPayload.softShadow, AMBIENT_INTENSITY, 1.);
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

#endif