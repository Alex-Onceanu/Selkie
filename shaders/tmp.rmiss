#version 460
#extension GL_EXT_ray_tracing : enable
layout(location = 0) rayPayloadInEXT vec3 ResultColor;

void main()
{
    ResultColor = vec3(0.2, 0.2, 0.2);
}
