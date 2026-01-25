#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_debug_printf : enable

struct Edit {
    vec3 pos;
    int type;
    float scale;
};

layout(set = 0, binding = 2, std430) buffer ssbo_t {
    Edit edits[];
} ssbo;

#define MAX_MERGES 8
struct payload_t {
    float time;
    vec3 hitColor;
    float minDist;
    float distToMinDist;
    float tHit;
    int nbHits;
    int hitIds[MAX_MERGES];
};

layout(location = 0) rayPayloadInEXT payload_t payload;

float sdfSphere(vec3 p, vec3 center, float radius)
{
    return length(p - center) - radius;
}

float sdfPlane(vec3 p)
{
    return p.y;
}

// returns signed distance from p to the closest edit in edits[payload.hitIds[...]] 
float map(vec3 p, int which)
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

void background()
{
    payload.hitColor = sky(gl_WorldRayDirectionEXT);
}

mat2 rot2D(float theta)
{
    return mat2(vec2(cos(theta), -sin(theta)), vec2(sin(theta), cos(theta)));
}

// simulating a closest hit shader here, since we can't call one from rmiss
void closestHit(vec3 p, float mixCoef)
{
    vec2 uv = p.xz;
    uv = fract(0.25 * uv);
    vec2 c = step(0.5, uv);
    vec3 checkerboard = ((step(1.0, c.x + c.y) - step(2.0, c.x + c.y)) * vec3(0.7) + vec3(0.3));

    vec3 lightPos = vec3(-0.5, 4., -7.);
    lightPos.xz *= rot2D(-2. * payload.time);
    vec3 normal = normalize(p - vec3(0., 1., 0.));
    vec3 sphClr = vec3(0.5, 0.3, 0.7) * max(0.2, dot(normalize(lightPos - p), normal)); 

    payload.hitColor = mix(checkerboard, sphClr, mixCoef);
}

void main()
{
    if(payload.nbHits <= 0)
    {
        background();
        return;
    }

    const int MAX_IT = 400;
    const float too_close = 0.000062;
    const float too_far = 1000.;
    vec3 p = gl_WorldRayOriginEXT;
    vec3 rd = normalize(gl_WorldRayDirectionEXT);
    float t = 0.;
    for(int i = 0; i < MAX_IT; i++)
    {
        float safeDist = 1. / 0.;
        float mixCoef = 0.0;
        if(payload.nbHits > 1)
        {
            for(int e = 0; e < payload.nbHits; e++)
            {
                vec2 merge = smin(safeDist, map(p, e));
                safeDist = merge.x;
                mixCoef = 1. - merge.y;
            }
        }
        else
        {
            safeDist = map(p, 0);
            mixCoef = ssbo.edits[payload.hitIds[0]].type == 0 ? 1. : 0.;
        }
        // debugPrintfEXT("Safe dist : %f", safeDist);

        p += rd * safeDist;
        t += safeDist;

        if(safeDist < payload.minDist)
        {
            payload.distToMinDist = length(p - gl_WorldRayOriginEXT) / length(gl_WorldRayOriginEXT);
            payload.minDist = safeDist;
        }

        if(t > too_far)
        {
            background();
            return;
        }

        if(safeDist <= too_close)
        {
            payload.tHit = length(p - gl_WorldRayOriginEXT) / length(gl_WorldRayOriginEXT);
            closestHit(p, mixCoef);
            return;
        }
    }
    background();
}