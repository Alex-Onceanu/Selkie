#version 460
#extension GL_EXT_ray_tracing : require

#include "main_payload.glsl"

layout(location = 0) rayPayloadInEXT payload_t payload;

void main()
{
    payload.shadow = 1.;
    payload.hitColor = mix(vec3(0.3, 0.5, 0.9), vec3(0.9), abs(normalize(gl_WorldRayDirectionEXT).y));
}