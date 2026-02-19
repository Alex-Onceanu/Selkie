#include <iostream>
#include <chrono>

#include "math.hpp"
#include "renderer.hpp"
#include "editor.hpp"

int main()
{
    try
    {
        auto window = sk::initWindow(1366, 768);
        auto editor = new Editor(window);

        auto startTime = std::chrono::high_resolution_clock::now();
        auto prevTime = startTime;

        int selected = sk::edit::add(0); // TODO : which edit is currently being modified?

        sk::math::Quaternion rot(1., sk::math::vec3(0.));
        rot.normalized();
        sk::math::vec3  inpos(5., 1.5, 0.);
        sk::math::vec3  albedo = sk::math::vec3(0.3, 1., 0.4);
        sk::math::vec3  dimensions = sk::edit::getDimensions(selected);
        sk::math::vec3  elongation{ 0. };
        float           roughness = 0.5;
        float           rounding = 0.;
        float           scale = 1.;
        float           onion = 0.;
        float           blendStrength = 9.;
        bool            negative = false;
        float           bend = 0.;
        float           norm = 2.;
        float           twist = 0.;
        
        editor->bind(sk::key::R, &rot);
        editor->bind(sk::key::P, &inpos, sk::math::vec2(-4., 12.));
        editor->bind(sk::key::A, &albedo, sk::math::vec2(0., 1.));
        editor->bind(sk::key::D, &dimensions, sk::math::vec2(0.01, 5.));
        editor->bind(sk::key::E, &elongation, sk::math::vec2(0., 3.));
        editor->bind(sk::key::M, &roughness, sk::math::vec2(0., 1.));
        editor->bind(sk::key::C, &rounding, sk::math::vec2(0., 2.));
        editor->bind(sk::key::S, &scale, sk::math::vec2(0.1, 2.));
        editor->bind(sk::key::O, &onion, sk::math::vec2(0., 10.));
        editor->bind(sk::key::K, &blendStrength, sk::math::vec2(4., 18.));
        editor->bind(sk::key::B, &bend, sk::math::vec2(0., 14.));
        editor->bind(sk::key::N, &norm, sk::math::vec2(0.2, 14.));
        editor->bind(sk::key::T, &twist, sk::math::vec2(0., 14.));
        editor->bind(sk::key::H, &negative);

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

            int digitPressed = -1;
            editor->update(digitPressed);

            if(digitPressed >= 0 and digitPressed <= 5)
            {
                selected = sk::edit::add(digitPressed);

                rot = sk::math::Quaternion(1., sk::math::vec3(0.));
                inpos = sk::math::vec3(5., 1., 0.);
                albedo = sk::math::vec3(0.5);
                dimensions = sk::edit::getDimensions(selected);
                elongation = sk::math::vec3(0.);
                roughness = 0.5;
                rounding = 0.;
                scale = 1.;
                onion = 0.;
                blendStrength = 9.;
                negative = false;
                bend = 0.;
                norm = 2.;
                twist = 0.;
            }

            sk::edit::setPos(selected, inpos);
            sk::edit::setRotation(selected, rot.toMatrix());
            sk::edit::setAlbedo(selected, albedo);
            sk::edit::setDimensions(selected, dimensions);
            sk::edit::setElongation(selected, elongation);
            sk::edit::setRoughness(selected, roughness);
            sk::edit::setRounding(selected, rounding);
            sk::edit::setScale(selected, scale);
            sk::edit::setOnion(selected, onion);
            sk::edit::setBlendStrength(selected, blendStrength);
            sk::edit::setBend(selected, bend);
            sk::edit::setNorm(selected, norm);
            sk::edit::setTwist(selected, twist);
            sk::edit::setNegative(selected, negative);

            sk::draw(elapsedTime);
            nbFrames++;
        }
        delete editor;
        sk::end();
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
