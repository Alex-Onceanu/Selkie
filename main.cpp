#include <iostream>
#include <chrono>

#include "math.hpp"
#include "renderer.hpp"

int main()
{
    auto startTime = std::chrono::high_resolution_clock::now();
    try
    {
        auto window = sk::initWindow(1366, 768);

        while(window->isAlive())
        {
            auto currentTime = std::chrono::high_resolution_clock::now();
            float elapsedTime = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
            
            sk::draw();
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
