#version 460
#extension GL_EXT_ray_tracing : require
// #extension GL_EXT_debug_printf : enable

#include "constants.glsl"
#include "main_payload.glsl"
#include "shadow_payload.glsl"
#include "ssbo.glsl"
#include "sdfs.glsl"

hitAttributeEXT vec3 normal;

layout(location = 0) rayPayloadInEXT payload_t payload;
layout(location = 1) rayPayloadEXT shadow_payload_t shadowPayload;

layout(set = 0, binding = 0) uniform accelerationStructureEXT bvh;

// computes partial derivative of exponential smooth minimum for each object then sums
material_t blendMaterial(const vec3 p, const float blend)
{
    material_t mat;

    edit_t ed = ssbo.edits[gl_PrimitiveID];
    float a_e = exp2(-sdf(p, gl_PrimitiveID) * blend);
    mat.albedo = ed.material.albedo * a_e;
    mat.roughness = ed.material.roughness * a_e;
    float sum = a_e;

    const int n = ssbo.edits[gl_PrimitiveID].nbNeighbours;
    for(int e = 0; e < n; e++)
    {
        ed = ssbo.edits[ssbo.edits[gl_PrimitiveID].neighbours[e]];
        a_e = exp2(-sdf(p, ssbo.edits[gl_PrimitiveID].neighbours[e]) * blend);

        mat.albedo += ed.material.albedo * a_e;
        mat.roughness += ed.material.roughness * a_e;
        sum += a_e;
    }

    mat.albedo /= sum;
    mat.roughness /= sum;
    return mat;
}

float shadowRay(const vec3 ro, const vec3 rd)
{
    shadowPayload.shadow = 1.;
    traceRayEXT(bvh, gl_RayFlagsNoneEXT, 0xFF, 1, 2, 1, ro + 0.01 * normal, T_MIN, rd, T_MAX, 1);

    return min(max(shadowPayload.shadow, AMBIENT_INTENSITY), 1.);
}

vec3 checkerboard(const vec3 p, const vec3 rd)
{
    vec2 uv = (gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT / length(gl_WorldRayDirectionEXT)).xz;
    uv = fract(uv);
    vec2 c = step(0.5, uv);
    return (step(1.0, c.x + c.y) - step(2.0, c.x + c.y)) * vec3(0.7) + vec3(0.3);
}

vec3 sphereColor(const vec3 p, const vec3 rd, const material_t mat, const vec3 lightPos)
{
    // return 0.5 * normal + vec3(0.5);
    const vec3 toLight = normalize(lightPos - p);
    const float diffuse = max(AMBIENT_INTENSITY, dot(normal, toLight));

    if(ssbo.edits[gl_PrimitiveID].glass)
    {
        payload.refractRay = true;
        payload.refr_ro = p + 0.001 * normal;
        payload.refr_rd = refract(normalize(rd), normal * payload.negativeRay, 1.04 + payload.negativeRay * 0.29);
        if(payload.negativeRay < 0.) return vec3(1., 0., 1.);
    }
    const float shadow = shadowRay(p, toLight);
    if(mat.roughness < 0.95)
    {
        payload.mirrorRay = true;
        payload.mir_ro = p + 0.001 * normal;
        payload.mir_rd = reflect(rd, normal);
        payload.mir_rough = mat.roughness;
    }
    payload.shadow = min(shadow, diffuse);
    return mat.albedo.x < 0. ? checkerboard(p, rd) : mat.albedo;
}

void main()
{
    const vec3 p = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT / length(gl_WorldRayDirectionEXT);
    vec3 lp = LIGHTPOS;

    // payload.hitColor = vec3(1.,0.,0.);
    // return;
    float blend = ssbo.edits[gl_PrimitiveID].blendStrength;
    for(int e = 0; e < ssbo.edits[gl_PrimitiveID].nbNeighbours; e++)
    {
        blend = max(blend, ssbo.edits[ssbo.edits[gl_PrimitiveID].neighbours[e]].blendStrength);
    }

    // lp.xz *= rot2D(-2.7 * time);
    payload.hitColor = sphereColor(p, gl_WorldRayDirectionEXT, blendMaterial(p, blend), lp);
}