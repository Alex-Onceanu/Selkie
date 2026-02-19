#include <iostream>
#include <chrono>

#include "math.hpp"
#include "renderer.hpp"
#include "editor.hpp"
#include "sph.hpp"

int main()
{
    try
    {
        srand(std::time(nullptr));
        auto window = sk::initWindow(1366, 768);
        auto editor = new Editor(window);

        std::vector<sk::math::vec3>* pparticles;
        sph::getHandles(&pparticles, nullptr);

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
        bool            glass = false;
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
        editor->bind(sk::key::G, &glass);
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

            // std::cout << "Update go " << std::endl;
            auto a = glfwGetTime();
            sph::update(elapsedTime);
            auto b = glfwGetTime();
            // std::cout << "Time for sph update : " << (b - a) * 1000.f << "ms" << std::endl;
            // std::cout << "Set pos go " << std::endl;
            for(int i = 0; i < pparticles->size(); ++i)
            {
                sk::edit::setPos(1 + i, (*pparticles)[i] + sk::math::vec3(5., 0.5, 0.));
            }
            // std::cout << "Set pos ok " << std::endl;

            int digitPressed = -1;
            editor->update(digitPressed);

            if(digitPressed >= 0 and digitPressed <= 5)
            {
                selected = sk::edit::add(digitPressed);

                rot = sk::math::Quaternion(1., sk::math::vec3(0.));
                inpos = sk::math::vec3(0., 1., 0.);
                albedo = sk::math::vec3(0.5);
                dimensions = sk::edit::getDimensions(selected);
                elongation = sk::math::vec3(0.);
                roughness = 0.5;
                rounding = 0.;
                scale = 1.;
                glass = false;
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
            sk::edit::setGlass(selected, glass);
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
