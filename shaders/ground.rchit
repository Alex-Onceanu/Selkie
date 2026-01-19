#version 460
#extension GL_EXT_ray_tracing : require

layout(set = 0, binding = 0) uniform accelerationStructureEXT bvh;

struct payload_t {
    float time;
    vec3 hitColor;
    float minDist;
};

layout(location = 0) rayPayloadInEXT payload_t payload;

mat2 rot2D(float theta)
{
    return mat2(vec2(cos(theta), -sin(theta)), vec2(sin(theta), cos(theta)));
}

void main()
{
    vec3 p = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    vec2 uv = p.xz;
    uv = fract(0.25 * uv);
    vec2 c = step(0.5, uv);
    vec3 lightPos = vec3(-0.5, 10., -10.);
    lightPos.xz *= rot2D(-payload.time);
    payload.minDist = 0.;
    traceRayEXT(bvh, gl_RayFlagsSkipClosestHitShaderEXT, 0xFF, 0, 1, 0, p, 1e-5, normalize(lightPos - p), 1. / 0., 0);
    float shadow = 1.;
    if(payload.minDist < 0.5)
    {
        shadow = 0.3;
    }
    payload.hitColor = shadow * ((step(1.0, c.x + c.y) - step(2.0, c.x + c.y)) * vec3(0.7) + vec3(0.3));
}