#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_debug_printf : enable

#include "util.glsl"
#include "sdfs.glsl"

hitAttributeEXT hitInfo_t hitInfo;

layout(push_constant) uniform PushConstants {
    float time;
    uint accumulationFrame;
};

layout(set = 0, binding = 2, std430) buffer e_ssbo_t {
    edit_t edits[];
} eSSBO;

layout(location = 0) rayPayloadInEXT payload_t payload;
layout(location = 1) rayPayloadEXT shadowPayload_t shadowPayload;


layout(set = 0, binding = 0) uniform accelerationStructureEXT bvh;

float rand(float co) { return fract(sin(co*(91.3458)) * 47453.5453); }

float randn(float co) {
    float u1 = rand(co);
    float u2 = rand(co + 1.0);
    return sqrt(-2.0 * log(u1)) * cos(6.28318530718 * u2);
}

/* void traceRayEXT(accelerationStructureEXT topLevel,
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
vec4 sphereColor(const vec3 p, const vec3 rd, const vec3 normal, const material_t mat, const float milkyness, const vec3 lightPos)
{
    vec3 albedo = mat.albedo;
    if(eSSBO.edits[gl_PrimitiveID].type == 1)
    {
        float checker = mod(floor(p.x * 2.0) + floor(p.z * 2.0), 2.0);
        if(checker < 1.0)
        {
            albedo = vec3(0.8, 0.2, 0.2);
        }
    }
    if(mat.roughness < 0.0)
    {
        return vec4(albedo, -mat.roughness);
    }
    if(payload.lifetime < 6)
    {
        vec3 randomDir = normalize(vec3(randn(p.x + time), randn(p.y + time), randn(p.z + time)) * 2.0 - 1.0);
        if(dot(normal, randomDir) < 0.0) randomDir = -randomDir;
        float isMilky = rand(p.x + p.y + p.z + time) > milkyness ? 1.0 : 0.0;
        vec3 newDir = mix(normal, randomDir, mat.roughness * isMilky);
        albedo = mix(albedo, vec3(1.0), milkyness);

        payload.lifetime++;
        traceRayEXT(bvh, gl_RayFlagsOpaqueEXT, 0xFF, 0, 1, 0, p + 0.001 * normal, T_MIN, newDir, T_MAX, 0);

        return vec4(payload.hitColor * albedo, payload.energy);  
    }
    else
    {
        return vec4(albedo, 0.0);
    }
}

void main()
{
    const vec3 p = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT / length(gl_WorldRayDirectionEXT);
    vec4 li = sphereColor(p, gl_WorldRayDirectionEXT, hitInfo.normal, eSSBO.edits[gl_PrimitiveID].material, eSSBO.edits[gl_PrimitiveID].milkyness, LIGHTPOS);
    payload.hitColor = li.rgb;
    payload.energy = li.a;
}