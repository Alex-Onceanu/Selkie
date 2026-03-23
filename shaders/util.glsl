#ifndef UTIL_GLSL
#define UTIL_GLSL

// _________________________________________CONSTANTS_________________________________________

#define FOV (70. * 3.1416 / 180.)
#define MAX_IT 200
#define T_MIN 1e-3
#define T_MAX 1e3
#define AMBIENT_INTENSITY 0.2
#define LIGHTPOS vec3(0.5, 12., 7.) // this should be an uniform
#define BLEND_STRENGTH 6.           // this too

// _________________________________________STRUCTS___________________________________________

struct payload_t {
    vec3 hitColor;
};

struct shadowPayload_t {
    float shadow;
};

struct material_t {
    vec3 albedo;
    float roughness;
};

struct edit_t {
    vec3 pos;
    int type;
    material_t material;
    vec3 scale;
    int _padding;
    // Should be aligned to 48 bytes (12 + 4 + 16 + 12 + 4 = 48)
};

struct hitInfo_t {
    vec3 normal;
    float shadow;
};

// _________________________________________FUNCTIONS___________________________________________


mat2 rot2D(const float theta)
{
    return mat2(vec2(cos(theta), -sin(theta)), vec2(sin(theta), cos(theta)));
}

#endif