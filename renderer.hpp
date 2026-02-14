#pragma once

#include <memory>

#include "math.hpp"
#include "window.hpp"
namespace sk
{
    // constructeur : initialise vulkan et renvoie la fenetre
    std::shared_ptr<sk::Window> initWindow(unsigned int width, unsigned int height);

    namespace Edits
    {
        unsigned int getNb();
        math::vec3 getPos(const unsigned int i);
        int getType(const unsigned int i);
        math::vec3 getAlbedo(const unsigned int i);
        float getRoughness(const unsigned int i);
        math::vec3 getDimensions(const unsigned int i);

        void setPos(const unsigned int i, const math::vec3 p);
        void setAlbedo(const unsigned int i, const math::vec3 a);
        void setRoughness(const unsigned int i, const float r);
        void setDimensions(const unsigned int i, const math::vec3 s);
    }

    // Appeler cette fonction 1 fois par frame max
    void draw(float t);

    void end();
};
