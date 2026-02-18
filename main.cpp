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

        sk::edit::add(1);
        sk::edit::setPos(1, sk::math::vec3(5.0, -2.3, 0.));
        sk::edit::setAlbedo(1, sk::math::vec3(0.5, 0.3, 0.7));
        sk::edit::setDimensions(1, sk::math::vec3(4.5, 0.75, 4.5));
        sk::edit::setNegative(0, false);
        sk::edit::setNegative(1, true);
        // sk::edit::setBlendStrength(0, 20);


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

            // sk::edit::setPos(0, sk::math::vec3(sinf(elapsedTime), 0.f, 0.f));

            // int i;
            // if(elapsedTime >= 4.f and not event)
            // {
            //     event = true;
            //     i = sk::edit::add(1);
            //     sk::edit::setDimensions(i, sk::math::vec3(1., 1., 1.));
            //     sk::edit::setPos(i, sk::math::vec3(3.6, 1.4, 0.));
            //     sk::edit::setAlbedo(i, sk::math::vec3(0., 1., 1.));
            //     sk::edit::setRoughness(i, 1.);
            //     sk::edit::setRotation(i, sk::math::Quaternion(sk::math::vec3(0., 1., 1.), 0.2 * elapsedTime).normalized().toMatrix());
            //     std::cout << "Event !" << std::endl;
            // }
            // if(elapsedTime >= 4.f)
            // {
            sk::edit::setRotation(0, sk::math::mat3::rotation(sk::math::vec3(0., 1., 1.), 0.7 * elapsedTime));
            sk::edit::setPos(0, sk::math::vec3(5., -0.5, 0.));
            // }

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
