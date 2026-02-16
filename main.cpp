#include <iostream>
#include <chrono>

#include "math.hpp"
#include "renderer.hpp"

int main()
{
    try
    {
        auto window = sk::initWindow(1366, 768);

        auto startTime = std::chrono::high_resolution_clock::now();
        auto prevTime = startTime;
        bool event = false;

        int nbFrames = 0;
        while(window->isAlive())
        {
            auto currentTime = std::chrono::high_resolution_clock::now();
            float elapsedTime = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
            float timeSinceLastSecond = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - prevTime).count();

            if(timeSinceLastSecond >= 1.0f)
            {
                std::cout << "FPS : " << nbFrames << std::endl;
                nbFrames = 0;
                prevTime = currentTime;
            }

            if(elapsedTime >= 4.f and not event)
            {
                event = true;
                auto i = sk::edit::add(1);
                sk::edit::setDimensions(i, sk::math::vec3(1., 1., 1.));
                sk::edit::setPos(i, sk::math::vec3(3.5, 1.3, 0.));
                sk::edit::setAlbedo(i, sk::math::vec3(0., 1., 1.));
                sk::edit::setRoughness(i, 1.);
                std::cout << "Event !" << std::endl;
            }

            sk::draw(elapsedTime);
            nbFrames++;
        }
        sk::end();
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
