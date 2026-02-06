#ifndef RAYMARCH_H
#define RAYMARCH_H

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
        mat.albedo += ssbo.edits[payload.hitIds[e]].material.albedo * a_e;
        mat.roughness += ssbo.edits[payload.hitIds[e]].material.roughness * a_e;
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


// found some nice random values at https://www.shadertoy.com/view/Xt23Ry
float rand(float co) { return fract(sin(co*(91.3458)) * 47453.5453); }
float rand(vec2 co){ return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453); }
float rand(vec3 co){ return rand(co.xy+rand(co.z)); }

// copy-pasted this, generates random points on the surface of a sphere
// iq's version of Keinert et al's inverse Spherical Fibonacci Mapping code
// https://www.shadertoy.com/view/lllXz4
vec2 inverseSF( vec3 p ) 
{
    float nbStars = 20000.;
    const float kTau = 6.28318530718;
    const float kPhi = (1.0+sqrt(5.0))/2.0;
    float kNum = nbStars;

    float k  = max(2.0, floor(log2(kNum*kTau*0.5*sqrt(5.0)*(1.0-p.z*p.z))/log2(kPhi+1.0)));
    float Fk = pow(kPhi, k)/sqrt(5.0);
    vec2  F  = vec2(round(Fk), round(Fk*kPhi)); // |Fk|, |Fk+1|
    
    vec2  ka = 2.0*F/kNum;
    vec2  kb = kTau*(fract((F+1.0)*kPhi)-(kPhi-1.0));    

    mat2 iB = mat2( ka.y, -ka.x, kb.y, -kb.x ) / (ka.y*kb.x - ka.x*kb.y);
    vec2 c = floor(iB*vec2(atan(p.y,p.x),p.z-1.0+1.0/kNum));

    float d = 8.0;
    float j = 0.0;
    for( int s=0; s<4; s++ ) 
    {
        vec2  uv = vec2(s&1,s>>1);
        float id = clamp(dot(F, uv+c),0.0,kNum-1.0); // all quantities are integers
        
        float phi      = kTau*fract(id*kPhi);
        float cosTheta = 1.0 - (2.0*id+1.0)/kNum;
        float sinTheta = sqrt(1.0-cosTheta*cosTheta);
        
        vec3 q = vec3( cos(phi)*sinTheta, sin(phi)*sinTheta, cosTheta );
        float tmp = dot(q-p, q-p);
        if( tmp<d ) 
        {
            d = tmp;
            j = id;
        }
    }
    return vec2( j, sqrt(d) );
}

vec3 skyColor(vec3 d)
{
    float starsDisplacement = 0.069;
    float starSize = 2000;
    float starSizeVariation = 300.;
    float starVoidThreshold = 0.249;
    float starFlickering = 1073;

    vec3 nd = normalize(d);
    vec2 centered = inverseSF(nd);
    float seed = centered.x;

    float rand1 = rand(seed);
    float rand2 = rand(rand1);
    float rand3 = rand(rand2);
    vec3 randVector = vec3(rand1, rand2, rand3);
    
    // cool mario galaxy background color : vec3(0.035, 0.114, 0.392)
    if(rand1 < starVoidThreshold) return vec3(0.); // so we have some void

    // second call to inverseSF because we needed to get the seed first
    // now we can use the seed to offset the stars for a more "natural" look
    vec2 a = inverseSF(normalize(nd + starsDisplacement * (-1. + 2. * randVector)));

    float dst = (starSize + starSizeVariation * rand1 + starFlickering * rand2 * pow(sin(3. * time * rand3), 5.)) * a.y;

    float glow = 1. / (0.001 + dst * dst);
    vec3 clr = 1. + 0.6 * randVector;
    float border = 1. - smoothstep(0.0, 0.015, centered.y); // temporary fix to the "neighbours" issue

    return tanh(glow * clr) * border;
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
    float rr = roughness;
    const vec3 mir = mirrorRay(p, normalize(reflect(rd, normal)), rr);

    return mix(mir, albedo, rr) * min(shadow, diffuse);
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

    float t = 0.;
    for(int i = 0; i < NB_IT; i++)
    {
        float safeDist = map(p);

        if(t > gl_RayTmaxEXT)
        {
            payload.hitColor = backgroundColor(p, rd, lp);
            return;
        }

        if(safeDist <= gl_RayTminEXT)
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
