#pragma once

#include <memory>

#include "math.hpp"
#include "window.hpp"

namespace sk
{
    namespace edit
    {
        unsigned int add(const int type__); // allocates a new edit and returns its index

        unsigned int getNb();
        math::vec3   getPos(        const unsigned int i);
        int          getType(       const unsigned int i);
        math::vec3   getAlbedo(     const unsigned int i);
        float        getRoughness(  const unsigned int i);
        math::vec3   getDimensions( const unsigned int i);

        void setPos(            const unsigned int i, const math::vec3 p);
        void setRotation(       const unsigned int i, const math::mat3 r);
        void setAlbedo(         const unsigned int i, const math::vec3 a);
        void setRoughness(      const unsigned int i, const float r);
        void setDimensions(     const unsigned int i, const math::vec3 s);
        void setRounding(       const unsigned int i, const float v);
        void setElongation(     const unsigned int i, const math::vec3 v);
        void setBlendStrength(  const unsigned int i, const float v);
        void setScale(          const unsigned int i, const float v);
        void setNegative(       const unsigned int i, const bool v);
        void setBend(           const unsigned int i, const float v);
        void setOnion(          const unsigned int i, const float v);
        void setNorm(           const unsigned int i, const float v);
        void setTwist(          const unsigned int i, const float v);
    }

    // constructeur : initialise vulkan et renvoie la fenetre
    std::shared_ptr<sk::Window> initWindow(unsigned int width, unsigned int height);

    // Appeler cette fonction 1 fois par frame max
    void draw(float t);

    void end();
};
