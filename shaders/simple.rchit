#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_debug_printf : enable

#include "constants.glsl"
#include "main_payload.glsl"
#include "edits_ssbo.glsl"
#include "sdfs.glsl"

#include "pbr.glsl"

layout(location = 0) rayPayloadInEXT payload_t payload;

// central differences
vec3 computeNormal(const vec3 p)
{
    const float eps = 1e-4;
    const vec2 h = vec2(eps,0);

    const int type = eSSBO.edits[gl_PrimitiveID].type;
    const vec3 objPos = eSSBO.edits[gl_PrimitiveID].pos;
    const vec3 objScale = eSSBO.edits[gl_PrimitiveID].scale;

    return normalize(vec3(sdf(p+h.xyy, type, objPos, objScale) - sdf(p-h.xyy, type, objPos, objScale),
                          sdf(p+h.yxy, type, objPos, objScale) - sdf(p-h.yxy, type, objPos, objScale),
                          sdf(p+h.yyx, type, objPos, objScale) - sdf(p-h.yyx, type, objPos, objScale)));
}

void main()
{
    payload.hitColor = vec3(1., 0., 0.);
    return;
    const vec3 p = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT / length(gl_WorldRayDirectionEXT);
    vec3 lp = LIGHTPOS;
    lp.xz *= rot2D(-2.7 * time);
    payload.hitColor = sphereColor(p, gl_WorldRayDirectionEXT, computeNormal(p), eSSBO.edits[gl_PrimitiveID].material, lp);
}