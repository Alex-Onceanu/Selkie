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

vec3 uniformRandomDirection(vec3 seed) {
    float u = rand(seed.x);
    float v = rand(seed.y);
    float theta = 2.0 * 3.14159265359 * u;
    float phi = acos(2.0 * v - 1.0);
    float x = sin(phi) * cos(theta);
    float y = sin(phi) * sin(theta);
    float z = cos(phi);
    return vec3(x, y, z);
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

    if(mat.roughness < -0.0001)
    {
        return vec4(albedo, -mat.roughness);
    }

    const int MAX_BOUNCES = 2;

    if(payload.lifetime >= MAX_BOUNCES)
    {
        return vec4(albedo, 0.0);
    }

    albedo = mix(albedo, vec3(1.0), milkyness);

    vec3 randomDir = uniformRandomDirection(payload.seed);
    if(dot(normal, randomDir) < 0.0) randomDir = -randomDir;

    float isMilky = fract(sin(dot(payload.seed.xy, vec2(12.9898, 78.233))) * 43758.5453) > milkyness ? 1.0 : 0.0;

    vec3 newDir = normalize(mix(normal, randomDir, mat.roughness * isMilky));

    payload.lifetime++;

    traceRayEXT(bvh, gl_RayFlagsOpaqueEXT, 0xFF, 0, 1, 0, p, 0.001, newDir, T_MAX, 0);
    vec3 finalColor = payload.hitColor * albedo;
    
    return vec4(finalColor, payload.energy);
}

void main()
{
    const vec3 p = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT / length(gl_WorldRayDirectionEXT);
    vec4 li = sphereColor(p, gl_WorldRayDirectionEXT, hitInfo.normal, eSSBO.edits[gl_PrimitiveID].material, eSSBO.edits[gl_PrimitiveID].milkyness, LIGHTPOS);
    payload.hitColor = li.rgb;
    payload.energy = li.a;
}