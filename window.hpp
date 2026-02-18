#pragma once

// changer cette ligne si cette classe est utilisee avec une autre API graphique que Vulkan
#define VULKAN_RENDERER

#ifdef VULKAN_RENDERER
#include <vulkan/vulkan.hpp>
#endif
#include <GLFW/glfw3.h>

#include <vector>
#include <tuple>

namespace sk
{
    enum class key
    {
        A = 65, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
        ESC = 256
    };

    // Abstraction de GLFW
    class Window
    {
    protected:
        GLFWwindow* window;
        
        public:
        Window(int __WINDOW_WIDTH, int __WINDOW_HEIGHT);
        ~Window();
        
        bool isAlive();
        void waitForEvent();
        
        GLFWwindow* getPtr();
        void getSize(int& pWidth, int& pHeight);
        void setResizeCallback(void (*__callback)(void));
        void setKeyEventCallback(void (*__callback)(sk::key, bool));
        
        std::pair<float, float> getMousePos();
        std::pair<bool, bool> getMouseClick();

        std::vector<const char*> getRequiredExtensions();

    #ifdef VULKAN_RENDERER
        vk::SurfaceKHR createSurface(VkInstance instance);
    #endif
    };
};
