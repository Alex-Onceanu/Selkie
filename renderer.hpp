#pragma once

#include <memory>

#include "math.hpp"
#include "window.hpp"
namespace sk
{
    // constructeur : initialise vulkan et renvoie la fenetre
    std::shared_ptr<sk::Window> initWindow(unsigned int width, unsigned int height);

    // Appeler cette fonction 1 fois par frame max
    void draw();

    void end();
};
