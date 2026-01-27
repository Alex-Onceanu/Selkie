#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_debug_printf : enable

struct edit_t {
    vec3 pos;
    int type;
    float scale;
};

#define MAX_MERGES 2
struct payload_t {
    vec3 hitColor;
    float time;
    int nbHits;
    int hitIds[MAX_MERGES];
    bool shadowRay;
    float softShadow;
    bool mirrorRay;
};

// if the ray intersects the smooth intersection between multiple objects, we need to shade them all and then interpolate
struct shadewho_t {
    int nb;
    int ids[MAX_MERGES];
    float ponderation[MAX_MERGES];
    // here add ponderation of each object instead of a single mix coef
};

layout(location = 0) rayPayloadInEXT payload_t payload;

layout(set = 0, binding = 0) uniform accelerationStructureEXT bvh;
layout(set = 0, binding = 2, std430) buffer ssbo_t {
    edit_t edits[];
} ssbo;

// _____________________________________________________Utility________________________________________________________

mat2 rot2D(const float theta)
{
    return mat2(vec2(cos(theta), -sin(theta)), vec2(sin(theta), cos(theta)));
}

// quadratic polynomial smooth minimum
vec2 smin(const float a, const float b)
{
    const float k = 0.25; // TODO : move this in cpu

    const float h = 1.0 - min( abs(a-b)/(4.0*k), 1.0 );
    const float w = h*h;
    const float m = w*0.5;
    const float s = w*k;
    return ((a<b) ? vec2(a-s,m) : vec2(b-s,1.0-m));
}

// returns a negative value if there is no intersection
float rayPlane(const vec3 ro, const vec3 rd)
{
    return -ro.y / rd.y;
}

// ________________________________________________________________________SDFs__________________________________________________________________

float sdfSphere(const vec3 p, const vec3 center, const float radius)
{
    return length(p - center) - radius;
}

// _______________________________________________________________________Scene__________________________________________________________________

// if there is an intersection with the infinite ground plane before t, updates p (inplace) and returns true
bool intersectsGround(const vec3 ro, const vec3 rd, const float t, out vec3 p)
{
    float tGround = rayPlane(ro, rd);
    if(tGround > gl_RayTminEXT && tGround < t)
    {
        p = ro + tGround * rd;
        return true;
    }
    return false;
}

// returns signed distance from p to the closest edit_t in edits[payload.hitIds[...]] 
float whichSdf(const vec3 p, const int which)
{
    switch(ssbo.edits[payload.hitIds[which]].type)
    {
    case 0:
        return sdfSphere(p, ssbo.edits[payload.hitIds[which]].pos + vec3(0., sin(payload.time), 0.), ssbo.edits[payload.hitIds[which]].scale);
    default:
        break;
    }
    return 1. / 0.;
}

// returns distance to the closest object in the scene
float map(const vec3 p)
{
    vec2 ans = vec2(1. / 0., 0.5);

    for(int e = 0; e < payload.nbHits; e++)
    {
        ans = smin(ans.x, whichSdf(p, e));
    }
    return ans.x;
}

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

float shadowRay(in vec3 ro, const vec3 rd)
{
    return 1.;
    if(payload.shadowRay) return 0.;

    float shadow = 1.;
    ro += 0.1 * rd;
    payload.shadowRay = true;
    payload.softShadow = 1.;
    payload.nbHits = 0;

    traceRayEXT(bvh, gl_RayFlagsSkipClosestHitShaderEXT, 0xFF, 0, 1, 0, ro, 1e-5, rd, 1. / 0., 0);
    shadow = clamp(payload.softShadow, 0.4, 1.);
    return shadow;
}

vec3 mirrorRay(in vec3 ro, const vec3 rd)
{
    return vec3(0.);
    if(payload.mirrorRay) return vec3(0.);

    ro += 0.1 * rd;
    payload.nbHits = 0;
    payload.mirrorRay = true;
    payload.shadowRay = false;
    
    traceRayEXT(bvh, gl_RayFlagsSkipClosestHitShaderEXT, 0xFF, 0, 1, 0, ro, 1e-5, rd, 1. / 0., 0);
    return payload.hitColor;
}

// ___________________________________________________________________Shading_________________________________________________________________________

vec3 skyColor(const vec3 rd)
{
    return mix(vec3(0.3, 0.5, 0.9), vec3(0.9), abs(rd.y));
}

vec3 groundColor(const vec3 p, const vec3 rd, const vec3 lightPos)
{
    vec2 uv = p.xz;
    uv = fract(0.25 * uv);
    const vec2 c = step(0.5, uv);
    const vec3 checkerboard = ((step(1.0, c.x + c.y) - step(2.0, c.x + c.y)) * vec3(0.7) + vec3(0.3));

    const vec3 toLight = normalize(lightPos - p);
    const vec3 normal = vec3(0., 1., 0.);
    const float diffuse = max(0.4, dot(normal, toLight));

    const float shadow = shadowRay(p, toLight);

    return checkerboard * min(shadow, diffuse);
}

// outputs either ground or sky color
vec3 backgroundColor(in vec3 p, const vec3 rd, const vec3 lightPos)
{
    if(intersectsGround(gl_WorldRayOriginEXT, rd, gl_RayTmaxEXT, p))
        return groundColor(p, rd, lightPos);
    else
        return skyColor(rd);
}

vec3 sphereColor(const vec3 p, const vec3 rd, vec3 albedo, const float metallicness, const vec3 lightPos)
{
    const vec3 toLight = normalize(lightPos - p);
    const vec3 normal = computeNormal(p);
    const float diffuse = max(0.4, dot(normal, toLight));

    const float shadow = shadowRay(p, toLight);

    if(!payload.mirrorRay)
    {
        const vec3 refl = normalize(reflect(rd, normal));
        albedo = mix(albedo, mirrorRay(p, refl), metallicness);
    }

    return albedo * min(shadow, diffuse);
}

// simulating a closest hit shader here, since we can't call one from rmiss
// so computes the color at position p
vec3 sceneColor(in vec3 p, const vec3 rd, const float t, const vec3 lightPos, const shadewho_t whom)
{
    // if(payload.shadowRay) return vec3(0.);
    // if(intersectsGround(gl_WorldRayOriginEXT, rd, t, p)) return groundColor(p, rd, lightPos);

    vec3 clr = vec3(0., 0., 0.);
    for(int i = 0; i < whom.nb; i++)
    {
        // TODO : stop hard-coding material info
        // use ssbo.edits[payload.hitIds[whom.ids[i]]].material
        clr += sphereColor(p, rd, vec3(0.5, 0.3, 0.7), 0.5, lightPos) * whom.ponderation[i];
    }
    return clr;
}

// ___________________________________________________________________Main_________________________________________________________________________

#define MAX_IT 200
#define shadowSoftness 16.

void main()
{
    vec3 p = gl_WorldRayOriginEXT;
    vec3 rd = normalize(gl_WorldRayDirectionEXT);

    vec3 lightPos = vec3(0.5, 6., 2.); // this should be an uniform
    lightPos.xz *= rot2D(-2.7 * payload.time);

    if(payload.nbHits <= 0)
    {
        payload.hitColor = backgroundColor(p, rd, lightPos);
        return;
    }

    float t = 0.1; p += t * rd;  // do not change this !
    for(int i = 0; i < MAX_IT; i++)
    {
        float safeDist = map(p);

        payload.softShadow = min(payload.softShadow, shadowSoftness * safeDist / t);

        p += rd * safeDist;
        t += safeDist;

        if(t > gl_RayTmaxEXT)
        {
            payload.hitColor = backgroundColor(p, rd, lightPos);
            return;
        }

        if(safeDist <= gl_RayTminEXT)
        {
            shadewho_t whom = computeShadeWho(p);
            // shadewho_t whom;
            // whom.nb = 1;
            // whom.ids[0] = 0;
            // whom.ponderation[0] = 1.;
            payload.hitColor = sceneColor(p, rd, t, lightPos, whom);
            return;
        }
    }
    payload.hitColor = backgroundColor(p, rd, lightPos);
}