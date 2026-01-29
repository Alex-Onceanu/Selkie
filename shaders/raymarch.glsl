#ifndef RAYMARCH_H
#define RAYMARCH_H

// if the ray intersects the smooth intersection between multiple objects, we need to shade them all and then interpolate
struct shadewho_t {
    int nb;
    int ids[MAX_MERGES];
    float ponderation[MAX_MERGES];
};

layout(push_constant) uniform PushConstants {
    float time;
};

#include "sdfs.glsl"

shadewho_t computeShadeWho(const vec3 p)
{
    shadewho_t ans;
    ans.nb = 0;

    vec2 acc = vec2(1. / 0., 0.5);
    for(int e = 0; e < payload.nbHits; e++)
    {
        acc = smin(acc.x, whichSdf(p, e));
        if(acc.x > gl_RayTminEXT) continue;

        ans.ids[ans.nb] = e;
        ans.ponderation[ans.nb] = acc.y;
        for(int j = 0; j < ans.nb; j++)
        {
            ans.ponderation[j] *= (1. - acc.y);
        }
        ans.nb++;
    }

    return ans;
}

// SDF central differences based on "mapNoShade" function
vec3 computeNormal(const vec3 p)
{
    const float eps = 1e-4;
    const vec2 h = vec2(eps,0);
    return normalize( vec3(map(p+h.xyy).x - map(p-h.xyy).x,
                           map(p+h.yxy).x - map(p-h.yxy).x,
                           map(p+h.yyx).x - map(p-h.yyx).x ) );
}

// ___________________________________________________________________Recursion_________________________________________________________________________

float shadowRay(const vec3 ro, const vec3 rd)
{
    shadowPayload.nbHits = 0;
    shadowPayload.softShadow = 1.;
    traceRayEXT(bvh, gl_RayFlagsSkipClosestHitShaderEXT, 0xFF, 1, 0, 1, ro + 0.1 * rd, T_MIN, rd, T_MAX, 1);

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

vec3 sphereColor(const vec3 p, const vec3 rd, const int which, const vec3 lightPos)
{
    const vec3 albedo = ssbo.edits[which].clr;
    const float roughness = ssbo.edits[which].roughness;
    const vec3 toLight = normalize(lightPos - p);
    const vec3 normal = computeNormal(p);
    const float diffuse = max(AMBIENT_INTENSITY, dot(normal, toLight));

    const float shadow = shadowRay(p, toLight);
    const vec3 mir = mirrorRay(p, normalize(reflect(rd, normal)));

    return mix(mir, albedo, roughness) * min(shadow, diffuse);
}

// simulating a closest hit shader here, since we can't call one from rmiss
// so computes the color at position p
vec3 sceneColor(in vec3 p, const vec3 rd, const float t, const vec3 lightPos, const shadewho_t whom)
{
    {
        vec3 pp = p; // wtf ? without this variable the gpu explodes
        if(intersectsGround(gl_WorldRayOriginEXT, rd, t, pp))
            return groundColor(pp, rd, lightPos);
    }

    // debugPrintfEXT("0 : pos = %f %f %f | type = %d | scale = %f | clr = %f %f %f | roughness = %f\n1 : pos = %f %f %f | type = %d | scale = %f | clr = %f %f %f | roughness = %f", ssbo.edits[0].pos.x, ssbo.edits[0].pos.y, ssbo.edits[0].pos.z, ssbo.edits[0].type, ssbo.edits[0].scale, ssbo.edits[0].clr.x, ssbo.edits[0].clr.y, ssbo.edits[0].clr.z, ssbo.edits[0].roughness, ssbo.edits[1].pos.x, ssbo.edits[1].pos.y, ssbo.edits[1].pos.z, ssbo.edits[1].type, ssbo.edits[1].scale, ssbo.edits[1].clr.x, ssbo.edits[1].clr.y, ssbo.edits[1].clr.z, ssbo.edits[1].roughness);

    vec3 clr = vec3(0., 0., 0.);
    {
        int nn = whom.nb;
        for(int i = 0; i < nn; i++)
        {
            float ff = whom.ponderation[i]; // same with this variable here
            clr += sphereColor(p, rd, payload.hitIds[whom.ids[i]], lightPos) * ff;
        }
    }
    return clr;
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

        p += rd * safeDist;
        t += safeDist;

        if(t > gl_RayTmaxEXT)
        {
            payload.hitColor = backgroundColor(p, rd, lp);
            return;
        }

        if(safeDist <= gl_RayTminEXT)
        {
            const shadewho_t sw = computeShadeWho(p);
            payload.hitColor = sceneColor(p, rd, t, lp, sw);
            return;
        }
    }
    payload.hitColor = backgroundColor(p, rd, lp);
}
#endif
