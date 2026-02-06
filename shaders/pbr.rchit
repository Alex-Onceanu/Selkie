#version 460
#extension GL_EXT_ray_tracing : require
#include "constants.glsl"
#include "main_payload.glsl"
#include "ssbo.glsl"
#include "sdfs.glsl"

hitAttributeEXT vec3 normal;

layout(location = 0) rayPayloadInEXT payload_t payload;
layout(set = 0, binding = 0) uniform accelerationStructureEXT bvh;

// computes partial derivative of exponential smooth minimum for each object then sums
material_t blendMaterial(const vec3 p)
{
    material_t mat;

    edit_t ed = ssbo.edits[gl_PrimitiveID];
    float a_e = exp2(-sdf(p, ed.type, ed.pos, ed.scale) * BLEND_STRENGTH);
    mat.albedo = ed.material.albedo * a_e;
    mat.roughness = ed.material.roughness * a_e;
    float sum = a_e;

    const int n = ssbo.edits[gl_PrimitiveID].nbNeighbours;
    for(int e = 0; e < n; e++)
    {
        ed = ssbo.edits[ssbo.edits[gl_PrimitiveID].neighbours[e]];
        a_e = exp2(-sdf(p, ed.type, ed.pos, ed.scale) * BLEND_STRENGTH);

        mat.albedo += ed.material.albedo * a_e;
        mat.roughness += ed.material.roughness * a_e;
        sum += a_e;
    }

    mat.albedo /= sum;
    mat.roughness /= sum;
    return mat;
}

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

vec3 sphereColor(const vec3 p, const vec3 rd, const material_t mat, const vec3 lightPos)
{
    const vec3 toLight = normalize(lightPos - p);
    const float diffuse = max(AMBIENT_INTENSITY, dot(normal, toLight));

    const float shadow = shadowRay(p, toLight);
    const vec3 mir = mirrorRay(p, reflect(rd, normal));

    return mix(mir, mat.albedo, mat.roughness) * min(shadow, diffuse);
}

void main()
{
    const vec3 p = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    vec3 lp = LIGHTPOS;
    lp.xz *= rot2D(-2.7 * time);
    payload.hitColor = sphereColor(p, gl_WorldRayDirectionEXT, blendMaterial(p), lp);
}