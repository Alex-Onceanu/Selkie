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

#define MAX_MERGES 8
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

// _______________________________________________________Sky__________________________________________________________


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

vec3 sky(vec3 d)
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

    float dst = (starSize + starSizeVariation * rand1 + starFlickering * rand2 * pow(sin(3. * payload.time * rand3), 5.)) * a.y;

    float glow = 1. / (0.001 + dst * dst);
    vec3 clr = 1. + 0.6 * randVector;
    float border = 1. - smoothstep(0.0, 0.015, centered.y); // temporary fix to the "neighbours" issue

    return tanh(glow * clr) * border;
}

// ________________________________________________________________________SDFs__________________________________________________________________

float sdfSphere(vec3 p, vec3 center, float radius)
{
    return length(p - center) - radius;
}

float sdfPlane(vec3 p)
{
    return p.y;
}


// _______________________________________________________________________Scene__________________________________________________________________

// returns signed distance from p to the closest edit in edits[payload.hitIds[...]] 
float whichSdf(vec3 p, int which)
{
    switch(ssbo.edits[payload.hitIds[which]].type)
    {
    case 0:
        return sdfSphere(p, ssbo.edits[payload.hitIds[which]].pos + vec3(0., sin(payload.time), 0.), ssbo.edits[payload.hitIds[which]].scale);
    case 1:
        return sdfPlane(p);
    default:
        break;
    }
    return 1. / 0.;
}

vec2 map(vec3 p)
{
    vec2 ans = vec2(1. / 0., 0.5);
    if(payload.nbHits > 1)
    {
        for(int e = 0; e < payload.nbHits; e++)
        {
            ans = smin(ans.x, whichSdf(p, e));
        }
    }
    else
    {
        ans = vec2(whichSdf(p, 0).x, float(ssbo.edits[payload.hitIds[0]].type));
    }
    return ans;
}

// SDF central differences based on "map" function
vec3 computeNormal(vec3 p)
{
    const float eps = 1e-4;
    const vec2 h = vec2(eps,0);
    return normalize( vec3(map(p+h.xyy).x - map(p-h.xyy).x,
                           map(p+h.yxy).x - map(p-h.yxy).x,
                           map(p+h.yyx).x - map(p-h.yyx).x ) );
}

// ___________________________________________________________________Main_________________________________________________________________________

void background()
{
    if(payload.shadowRay) return;
    payload.hitColor = sky(gl_WorldRayDirectionEXT);
}

float shadowRay(in vec3 ro, in vec3 rd)
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

vec3 mirrorRay(in vec3 ro, in vec3 rd)
{
    if(payload.mirrorRay) return vec3(0.);

    ro += 0.1 * rd;
    payload.nbHits = 0;
    payload.mirrorRay = true;
    payload.shadowRay = false;
    
    traceRayEXT(bvh, gl_RayFlagsSkipClosestHitShaderEXT, 0xFF, 0, 1, 0, ro, 1e-5, rd, 1. / 0., 0);
    return payload.hitColor;
}

// simulating a closest hit shader here, since we can't call one from rmiss
// so computes the color at position p
// problem : how to only compute the color of the object this ray is currently hitting ?
void closestHit(vec3 p, vec3 rd, float mixCoef)
{
    if(payload.shadowRay) return;

    vec2 uv = p.xz;
    uv = fract(0.25 * uv);
    vec2 c = step(0.5, uv);
    vec3 checkerboard = ((step(1.0, c.x + c.y) - step(2.0, c.x + c.y)) * vec3(0.7) + vec3(0.3));
    vec3 sphClr = vec3(0.5, 0.3, 0.7); 

    vec3 lightPos = vec3(0.5, 6., 2.);
    lightPos.xz *= rot2D(-2.7 * payload.time);
    vec3 toLight = normalize(lightPos - p);
    vec3 normal = computeNormal(p);
    float diffuse = max(0.4, dot(normal, toLight));
    float shadow = shadowRay(p, toLight);
    float metallicness = 0.6;

    if(!payload.mirrorRay)
    {
        vec3 refl = normalize(reflect(rd, normal));
        // debugPrintfEXT("rd : %f %f %f, normal : %f %f %f, refl : %f %f %f", rd.x, rd.y, rd.z, normal.x, normal.y, normal.z, refl.x, refl.y, refl.z);
        sphClr = mix(sphClr, mirrorRay(p, refl), metallicness); // sends a mirror ray even when not touching the sphere...
    }

    payload.hitColor = mix(sphClr, checkerboard, mixCoef) * min(shadow, diffuse);
}

void main()
{
    if(payload.nbHits <= 0)
    {
        background();
        return;
    }

    const int MAX_IT = 400; // TODO : accelerate ground SDF so that we can reduce this
    const float too_close = 0.0001;
    const float too_far = 1000.;
    float rayLength = length(gl_WorldRayDirectionEXT);
    float shadowCoef = 16.;

    vec3 p = gl_WorldRayOriginEXT;
    vec3 rd = normalize(gl_WorldRayDirectionEXT);
    float t = 0.1;
    p += t * rd;
    for(int i = 0; i < MAX_IT; i++)
    {
        vec2 mapped = map(p);
        float safeDist = mapped.x;
        float mixCoef = mapped.y;

        // if(safeDist < 0.) payload.softShadow = 0.;
        payload.softShadow = min(payload.softShadow, shadowCoef * safeDist / t);

        p += rd * safeDist;
        t += safeDist;

        if(t > too_far)
        {
            background();
            return;
        }

        if(safeDist <= too_close)
        {
            closestHit(p, rd, mixCoef);
            return;
        }
    }
    background();
}