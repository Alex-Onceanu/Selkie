#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec3 hitColor;

void main()
{
    vec2 uv = (gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT).xz;
    uv = fract(uv);
    vec2 c = step(0.5, uv);
    hitColor = (step(1.0, c.x + c.y) - step(2.0, c.x + c.y)) * vec3(0.7) + vec3(0.3);
}