#ifndef RAYMARCH_H
#define RAYMARCH_H

// if the ray intersects the smooth intersection between multiple objects, we need to shade them all and then interpolate
// for now we interpolate the material, and shade only once
// though this will have to be changed if one day we want custom materials with arbitrary shader execution
struct material_t {
    vec3 albedo;
    float roughness;
};

layout(push_constant) uniform PushConstants {
    float time;
};

#include "sdfs.glsl"

// computes partial derivative of exponential smooth minimum for each object then sums
material_t blendMaterial(const vec3 p)
{
    material_t mat;
    mat.albedo = vec3(0.);
    mat.roughness = 0.;

    float sum = 0.;
    for(int e = 0; e < payload.nbHits; e++)
    {
        float a_e = exp2(-whichSdf(p, e) * BLEND_STRENGTH);
        mat.albedo += ssbo.edits[payload.hitIds[e]].clr * a_e;
        mat.roughness += ssbo.edits[payload.hitIds[e]].roughness * a_e;
        sum += a_e;
    }

    mat.albedo /= sum;
    mat.roughness /= sum;
    return mat;
}

// central differences
vec3 computeNormal(const vec3 p)
{
    const float eps = 1e-4;
    const vec2 h = vec2(eps,0);
    return normalize( vec3(map(p+h.xyy) - map(p-h.xyy),
                           map(p+h.yxy) - map(p-h.yxy),
                           map(p+h.yyx) - map(p-h.yyx) ) );
}

// ___________________________________________________________________Recursion_________________________________________________________________________

float shadowRay(const vec3 ro, const vec3 rd)
{
    shadowPayload.nbHits = 0;
    shadowPayload.softShadow = 1.;
    traceRayEXT(bvh, gl_RayFlagsSkipClosestHitShaderEXT, 0xFF, 1, 0, 1, ro, T_MIN, rd, T_MAX, 1);

    return clamp(shadowPayload.softShadow, AMBIENT_INTENSITY, 1.);
}

// ___________________________________________________________________Shading_________________________________________________________________________

vec3 skyColor(const vec3 rd)
{
    return mix(vec3(0.3, 0.5, 0.9), vec3(0.9), abs(rd.y));
}

vec3 groundColor(const vec3 p, const vec3 rd, const vec3 lightPos)
{
    vec2 uv = fract(0.25 * p.xz);
    const vec2 c = step(0.5, uv);
    const vec3 checkerboard = ((step(1.0, c.x + c.y) - step(2.0, c.x + c.y)) * vec3(0.7) + vec3(0.3));

    const vec3 toLight = normalize(lightPos - p);
    const float diffuse = max(AMBIENT_INTENSITY, toLight.y);

    return checkerboard * min(shadowRay(p, toLight), diffuse);
}

// outputs either ground or sky color
vec3 backgroundColor(in vec3 p, const vec3 rd, const vec3 lightPos)
{
    if(intersectsGround(gl_WorldRayOriginEXT, rd, gl_RayTmaxEXT, p))
        return groundColor(p, rd, lightPos);
    else
        return skyColor(rd);
}

vec3 sphereColor(const vec3 p, const vec3 rd, const vec3 albedo, const float roughness, const vec3 lightPos)
{
    const vec3 toLight = normalize(lightPos - p);
    const vec3 normal = computeNormal(p);
    const float diffuse = max(AMBIENT_INTENSITY, dot(normal, toLight));

    const float shadow = shadowRay(p, toLight);
    const vec3 mir = mirrorRay(p, normalize(reflect(rd, normal)));

    return mix(mir, albedo, roughness) * min(shadow, diffuse);
}

// simulating a closest hit shader here, since we can't call one from rmiss
// so computes the color at position p
vec3 sceneColor(in vec3 p, const vec3 rd, const float t, const vec3 lightPos, const material_t mat)
{
    {
        vec3 pp = p; // wtf ? without this variable the gpu explodes
        if(intersectsGround(gl_WorldRayOriginEXT, rd, t, pp))
            return groundColor(pp, rd, lightPos);
    }

    return sphereColor(p, rd, mat.albedo, mat.roughness, lightPos);
}

// ___________________________________________________________________Main_________________________________________________________________________

void raymarch(const int NB_IT)
{
    
    vec3 p = gl_WorldRayOriginEXT;
    vec3 rd = normalize(gl_WorldRayDirectionEXT);
    vec3 lp = LIGHTPOS;
    lp.xz *= rot2D(-2.7 * time);

    if(payload.nbHits <= 0)
    {
        payload.hitColor = backgroundColor(p, rd, lp);
        return;
    }

    float t = 0.1; p += t * rd;  // do not change this !
    for(int i = 0; i < NB_IT; i++)
    {
        float safeDist = map(p);


        if(t > gl_RayTmaxEXT)
        {
            payload.hitColor = backgroundColor(p, rd, lp);
            return;
        }

        if(abs(safeDist) <= gl_RayTminEXT)
        {
            payload.hitColor = sceneColor(p, rd, t, lp, blendMaterial(p));
            return;
        }

        p += rd * safeDist;
        t += safeDist;
    }
    payload.hitColor = backgroundColor(p, rd, lp);
}
#endif
