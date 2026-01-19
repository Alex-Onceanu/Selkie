#version 460
#extension GL_EXT_ray_tracing : require

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
    vec3 lightPos = vec3(-0.5, 10., -10.);
    lightPos.xz *= rot2D(-payload.time);
    vec3 normal = normalize(p - vec3(0., 1., 0.));
    payload.hitColor = vec3(0.5, 0.3, 0.7) * max(0.2, dot(normalize(lightPos - p), normal)); 
}