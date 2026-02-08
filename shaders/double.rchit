#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_debug_printf : enable

#include "constants.glsl"
#include "main_payload.glsl"
#include "edits_ssbo.glsl"
#include "merges_ssbo.glsl"
#include "sdfs.glsl"

#include "pbr.glsl"

layout(location = 0) rayPayloadInEXT payload_t payload;

// returns signed distance to the smooth union of the two objects
float map(const vec3 p)
{
    const int i = mSSBO.merges[gl_PrimitiveID].first, j = mSSBO.merges[gl_PrimitiveID].second;
    return - (1. / BLEND_STRENGTH) * log2(exp2(-sdf(p, eSSBO.edits[i].type, eSSBO.edits[i].pos, eSSBO.edits[i].scale) * BLEND_STRENGTH)
                                        + exp2(-sdf(p, eSSBO.edits[j].type, eSSBO.edits[j].pos, eSSBO.edits[j].scale) * BLEND_STRENGTH));
}

float mixCoef(const vec3 p)
{
    const int i = mSSBO.merges[gl_PrimitiveID].first, j = mSSBO.merges[gl_PrimitiveID].second;

    const float a = exp2(-sdf(p, eSSBO.edits[i].type, eSSBO.edits[i].pos, eSSBO.edits[i].scale) * BLEND_STRENGTH);
    const float b = exp2(-sdf(p, eSSBO.edits[j].type, eSSBO.edits[j].pos, eSSBO.edits[j].scale) * BLEND_STRENGTH);
    
    return a / (a + b);
}

material_t blendMaterial(const vec3 p)
{
    const int i = mSSBO.merges[gl_PrimitiveID].first, j = mSSBO.merges[gl_PrimitiveID].second;
    float t = mixCoef(p);
    
    material_t mat;
    mat.albedo = mix(eSSBO.edits[j].material.albedo, eSSBO.edits[i].material.albedo, t);
    mat.roughness = mix(eSSBO.edits[j].material.roughness, eSSBO.edits[i].material.roughness, t);
    return mat;
}

// central differences
vec3 computeNormal(const vec3 p)
{
    const float eps = 1e-4;
    const vec2 h = vec2(eps,0);

    return normalize(vec3(map(p+h.xyy) - map(p-h.xyy),
                          map(p+h.yxy) - map(p-h.yxy),
                          map(p+h.yyx) - map(p-h.yyx)));
}

void main()
{
    const vec3 p = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT / length(gl_WorldRayDirectionEXT);
    vec3 lp = LIGHTPOS;
    lp.xz *= rot2D(-2.7 * time);
    payload.hitColor = sphereColor(p, gl_WorldRayDirectionEXT, computeNormal(p), blendMaterial(p), lp);
}