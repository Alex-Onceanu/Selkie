#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_debug_printf : enable

struct Edit {
    vec3 pos;
    int type;
    float scale;
};

layout(set = 0, binding = 0) uniform accelerationStructureEXT bvh;
layout(set = 0, binding = 2, std430) buffer ssbo_t {
    Edit edits[];
} ssbo;

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

layout(location = 0) rayPayloadInEXT payload_t payload;

// ____________________________________________Global variables & constants____________________________________________

#define GROUND_SHADE_ID -1

// if the ray intersects the smooth intersection between multiple objects, we need to shade them all and then interpolate
int nbShadeWho = 0;
int shadeWho[MAX_MERGES];
float shadeWhoPonderation[MAX_MERGES];
float shadeWhoPonderationSum = 0.;

vec3 lightPos = vec3(0.5, 6., 2.); // this should be an uniform

// _____________________________________________________Utility________________________________________________________

mat2 rot2D(float theta)
{
    return mat2(vec2(cos(theta), -sin(theta)), vec2(sin(theta), cos(theta)));
}

// quadratic polynomial smooth minimum
vec2 smin(float a, float b)
{
    float k = 0.25;

    float h = 1.0 - min( abs(a-b)/(4.0*k), 1.0 );
    float w = h*h;
    float m = w*0.5;
    float s = w*k;
    return ((a<b) ? vec2(a-s,m) : vec2(b-s,1.0-m));
}

// returns a negative value if there is no intersection
float rayPlane(vec3 ro, vec3 rd)
{
    return -ro.y / rd.y;
}

// ________________________________________________________________________SDFs__________________________________________________________________

float sdfSphere(vec3 p, vec3 center, float radius)
{
    return length(p - center) - radius;
}

// _______________________________________________________________________Scene__________________________________________________________________

// if there is an intersection with the infinite ground plane before t, updates p and shadeWho (inplace)
void intersectGround(in vec3 ro, in vec3 rd, in float t, out vec3 p)
{
    float tGround = rayPlane(ro, rd);
    if(tGround > gl_RayTminEXT && tGround < t)
    {
        p = ro + tGround * rd;
        nbShadeWho = 1;
        shadeWho[0] = GROUND_SHADE_ID;
        shadeWhoPonderation[0] = 1.;
        shadeWhoPonderationSum = 1.;
    }
}

// returns signed distance from p to the closest edit in edits[payload.hitIds[...]] 
float whichSdf(vec3 p, int which)
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

// returns distance to the closest object in the scene and computes shadeWho to know which material to use
float map(vec3 p)
{
    nbShadeWho = 0;
    vec2 ans = vec2(1. / 0., 0.5);

    for(int e = 0; e < payload.nbHits; e++)
    {
        ans = smin(ans.x, whichSdf(p, e));

        if(ans.y > 0.999)
        {
            // the ray intersects a single object
            nbShadeWho = 1;
            shadeWho[0] = e;
            shadeWhoPonderation[0] = 1.;
            shadeWhoPonderationSum = 1.;
        }
        else if(ans.y > 0.001)
        {
            // the ray intersects smin combination of previous objects and "e", so add it to shadeWho
            shadeWho[nbShadeWho] = e;
            shadeWhoPonderation[nbShadeWho] = ans.y;
            shadeWhoPonderationSum += ans.y;
            nbShadeWho++;
        }
        // else don't change anything
    }
    return ans.x;
}

// returns distance to the closest object in the scene
float mapNoShade(vec3 p)
{
    vec2 ans = vec2(1. / 0., 0.5);
    for(int e = 0; e < payload.nbHits; e++)
    {
        ans = smin(ans.x, whichSdf(p, e));
    }
    return ans.x;
}

// SDF central differences based on "mapNoShade" function
vec3 computeNormal(vec3 p)
{
    const float eps = 1e-4;
    const vec2 h = vec2(eps,0);
    return normalize( vec3(mapNoShade(p+h.xyy).x - mapNoShade(p-h.xyy).x,
                           mapNoShade(p+h.yxy).x - mapNoShade(p-h.yxy).x,
                           mapNoShade(p+h.yyx).x - mapNoShade(p-h.yyx).x ) );
}

// ___________________________________________________________________Recursion_________________________________________________________________________

float shadowRay(out vec3 ro, vec3 rd)
{
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

vec3 mirrorRay(vec3 ro, vec3 rd)
{
    if(payload.mirrorRay) return vec3(0.);

    ro += 0.1 * rd;
    payload.nbHits = 0;
    payload.mirrorRay = true;
    payload.shadowRay = false;
    
    traceRayEXT(bvh, gl_RayFlagsSkipClosestHitShaderEXT, 0xFF, 0, 1, 0, ro, 1e-5, rd, 1. / 0., 0);
    return payload.hitColor;
}

// ___________________________________________________________________Shading_________________________________________________________________________

vec3 backgroundColor(vec3 rd)
{
    return mix(vec3(0.3, 0.5, 0.9), vec3(0.9), abs(rd.y));
}

vec3 groundColor(vec3 p, vec3 rd)
{
    vec2 uv = p.xz;
    uv = fract(0.25 * uv);
    vec2 c = step(0.5, uv);
    vec3 checkerboard = ((step(1.0, c.x + c.y) - step(2.0, c.x + c.y)) * vec3(0.7) + vec3(0.3));

    vec3 toLight = normalize(lightPos - p);
    vec3 normal = vec3(0., 1., 0.);
    float diffuse = max(0.4, dot(normal, toLight));
    
    float shadow = shadowRay(p, toLight);

    return checkerboard * min(shadow, diffuse);
}

vec3 sphereColor(vec3 p, vec3 rd, vec3 albedo, float metallicness)
{
    vec3 toLight = normalize(lightPos - p);
    vec3 normal = computeNormal(p);
    float diffuse = max(0.4, dot(normal, toLight));

    float shadow = shadowRay(p, toLight);

    if(!payload.mirrorRay)
    {
        vec3 refl = normalize(reflect(rd, normal));
        albedo = mix(albedo, mirrorRay(p, refl), metallicness);
    }

    return albedo * min(shadow, diffuse);
}

// simulating a closest hit shader here, since we can't call one from rmiss
// so computes the color at position p
vec3 sceneColor(vec3 p, vec3 rd)
{
    if(payload.shadowRay) return vec3(0.);

    vec3 clr = vec3(0.);

    for(int i = 0; i < nbShadeWho; i++)
    {
        if(shadeWho[i] == GROUND_SHADE_ID) return groundColor(p, rd);

        // TODO : stop hard-coding material info
        // use ssbo.edits[payload.hitIds[shadeWho[i]]].material
        clr += sphereColor(p, rd, vec3(0.5, 0.3, 0.7), 0.5) * shadeWhoPonderation[i] / shadeWhoPonderationSum;
    }
    return clr;
}

// ___________________________________________________________________Main_________________________________________________________________________

void main()
{
    vec3 p = gl_WorldRayOriginEXT;
    vec3 rd = normalize(gl_WorldRayDirectionEXT);

    if(payload.nbHits <= 0)
    {
        payload.hitColor = backgroundColor(rd);
        return;
    }

    const int MAX_IT = 200;
    float shadowSoftness = 16.;
    lightPos.xz *= rot2D(-2.7 * payload.time);

    float t = 0.1; p += t * rd;  // do not change this !
    for(int i = 0; i < MAX_IT; i++)
    {
        float safeDist = map(p);

        payload.softShadow = min(payload.softShadow, shadowSoftness * safeDist / t);

        p += rd * safeDist;
        t += safeDist;

        if(t > gl_RayTmaxEXT)
        {
            payload.hitColor = backgroundColor(rd);
            intersectGround(gl_WorldRayOriginEXT, rd, t, p);
            return;
        }

        if(safeDist <= gl_RayTminEXT)
        {
            // intersectGround(gl_WorldRayOriginEXT, rd, t, p);
            // payload.hitColor = vec3(0., 1., 0.);
            // nbShadeWho = 1;
            // shadeWho[0] = 0;
            // shadeWhoPonderationSum = 1.;
            // shadeWhoPonderation[0] = 1.;
            payload.hitColor = sceneColor(p, rd);
            return;
        }
    }
    intersectGround(gl_WorldRayOriginEXT, rd, t, p);
    payload.hitColor = backgroundColor(rd);
}