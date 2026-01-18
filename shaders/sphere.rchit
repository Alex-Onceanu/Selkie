#version 460
#extension GL_EXT_ray_tracing : require

struct payload_t {
    float time;
    vec3 hitColor;
};

layout(location = 0) rayPayloadInEXT payload_t payload;

void main()
{
    vec3 p = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    vec3 lightDir = normalize(vec3(0.3, -0.6, -0.8));
    vec3 normal = normalize(p - vec3(0., 1., 0.));
    payload.hitColor = 0.5 * vec3(0.5, 0.3, 0.7) * max(0.2, dot(-lightDir, normal)); 
}