#version 460
#extension GL_EXT_ray_tracing : require
// #extension GL_EXT_debug_printf : enable

#include "shadow_payload.glsl"

layout(location = 1) rayPayloadInEXT shadow_payload_t payload;

void main()
{
    // rien mdr
}