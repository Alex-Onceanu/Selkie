#ifndef CONSTANTS_H
#define CONSTANTS_H

#define FOV (70. * 3.1416 / 180.)
#define MAX_MERGES 32
#define MAX_IT 256
#define SHADOW_MAX_IT 200
#define REFLECT_MAX_IT 200
#define T_MIN 1e-3
#define T_MAX 1e3
#define AMBIENT_INTENSITY 0.2
#define LIGHTPOS vec3(0.5, 12., 7.) // this should be an uniform
#define BLEND_STRENGTH 9.           // this too

#endif