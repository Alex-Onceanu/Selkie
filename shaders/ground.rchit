#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_debug_printf : enable

layout(set = 0, binding = 0) uniform accelerationStructureEXT bvh;

struct payload_t {
    float time;
    vec3 hitColor;
    float minDist;
    float distToMinDist;
    float tHit;
};

layout(location = 0) rayPayloadInEXT payload_t payload;

mat2 rot2D(float theta)
{
    return mat2(vec2(cos(theta), -sin(theta)), vec2(sin(theta), cos(theta)));
}

void main()
{
    // if(payload.tHit < gl_HitTEXT)
    // {
    //     // debugPrintfEXT("tHit < HitTEXT !! tHit : %f\n", gl_HitTEXT);
    //     // payload.hitColor = vec3(1., 0., 0.);
    //     return;
    // }
    vec3 p = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * payload.tHit;
    vec2 uv = p.xz;
    uv = fract(0.25 * uv);
    vec2 c = step(0.5, uv);
    vec3 lightPos = vec3(-0.5, 4., -7.);
    // lightPos.xz *= rot2D(-payload.time);
    
    payload.tHit = 1. / 0.;
    payload.minDist = 1. / 0.;
    float shadow = 1.;

    p += vec3(0., 0.01, 0.);
    traceRayEXT(bvh, gl_RayFlagsSkipClosestHitShaderEXT, 0xFF, 0, 1, 0, p, 1e-5, normalize(lightPos - p), 1. / 0., 0);
    
    float k = 16.;
    shadow = 0.3 + 0.7 * smoothstep(0., 1., k * payload.minDist / payload.distToMinDist);
    // if(payload.minDist < 1e3)
    // debugPrintfEXT("min dist : %f\n", k * payload.minDist / payload.distToMinDist);
    
    
    payload.hitColor = shadow * ((step(1.0, c.x + c.y) - step(2.0, c.x + c.y)) * vec3(0.7) + vec3(0.3));
}