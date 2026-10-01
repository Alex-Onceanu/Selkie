#pragma once

#include <memory>

#include "math.hpp"
#include "window.hpp"
namespace sk
{
    // constructeur : initialise vulkan et renvoie la fenetre
    std::shared_ptr<sk::Window> initWindow(unsigned int width, unsigned int height);

    // Appeler cette fonction 1 fois par frame max
    void draw(float t);

    /*

    "
        uniform vec3 a;
        uniform float x;

        float sdf(vec3 p)
        {
        
        }

        vec3 color(vec3 p, vec3 normal)
        {
        
        }
    "

    jaaj.sdf
    |
    | bash ?
    |
    v
    jaaj.glsl (a <- ssbo.edits[box.id + ? * ])
    |
    | glslc
    |
    v
    


    class Box
    {
    private:
        float x, y, z, w, h, d;
        int primitive_id;

    public:
        Box(x, y, z, w, h, d, sdf_filename)
        uniform(float )
        uniform(vec2)
        [...]

        setX(x)
        {
            bvh[id].transform[0][3] = x;
        }

        setW(w)
        {
            bvh.aabbs[id].w = w;
            should_rebuild_bvh = true;
        }

        [...]
    };

    b = Box([...]);
    c = Box([...]);
    init([b, c, ...]);
    while(1)
        b.uniform(...);
        c.uniform(...);
        draw(dt);
    */

    void end();
};
