#version 460
#extension GL_EXT_ray_tracing : require

struct payload_t {
    float time;
    vec3 hitColor;
};

layout(location = 0) rayPayloadInEXT payload_t payload;

void main()
{
    vec2 uv = (gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT).xz;
    uv = fract(0.25 * uv);
    vec2 c = step(0.5, uv);
    payload.hitColor = (step(1.0, c.x + c.y) - step(2.0, c.x + c.y)) * vec3(0.4) + vec3(0.1);
}