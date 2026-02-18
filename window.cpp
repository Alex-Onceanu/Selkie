#include "window.hpp"
#ifdef VULKAN_RENDERER
#include <vulkan/vulkan.hpp>
#endif
#include <GLFW/glfw3.h>


namespace
{
    void (*callback)(void);

    void framebufferResizeCallback(GLFWwindow* window, int width, int height)
    {
        (*callback)();
    }

    void (*keyEventCallback)(sk::key, bool);

    void glfwKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
    {
        (*keyEventCallback)(sk::key(key), action == GLFW_PRESS);
    }
}

sk::Window::Window(int __WINDOW_WIDTH, int __WINDOW_HEIGHT)
{
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    window = glfwCreateWindow(__WINDOW_WIDTH, __WINDOW_HEIGHT, "Selkie window", nullptr, nullptr);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    glfwSetWindowAspectRatio(window, __WINDOW_WIDTH, __WINDOW_HEIGHT);
    
    const int MIN_WIDTH = 160;
    const int MIN_HEIGHT = MIN_WIDTH * __WINDOW_HEIGHT / __WINDOW_WIDTH;
    glfwSetWindowSizeLimits(window, MIN_WIDTH, MIN_HEIGHT, GLFW_DONT_CARE, GLFW_DONT_CARE);
}

sk::Window::~Window()
{
    glfwDestroyWindow(window);
    glfwTerminate();
}

bool sk::Window::isAlive()
{
    glfwPollEvents();
    return not glfwWindowShouldClose(window);
}

void sk::Window::waitForEvent()
{
    glfwWaitEvents();
}

GLFWwindow* sk::Window::getPtr()
{
    return window;
}

void sk::Window::getSize(int& pWidth, int& pHeight)
{
    glfwGetFramebufferSize(window, &pWidth, &pHeight);
}

void sk::Window::setResizeCallback(void (*callback_)(void))
{
    callback = callback_;
}

void sk::Window::setKeyEventCallback(void (*__callback)(sk::key, bool))
{
    keyEventCallback = __callback;
    glfwSetKeyCallback(window, glfwKeyCallback);
}

std::pair<float, float> sk::Window::getMousePos()
{
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    return std::make_pair(xpos, ypos);
}

std::pair<bool, bool> sk::Window::getMouseClick()
{
    return std::make_pair(glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS, glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);
}

std::vector<const char*> sk::Window::getRequiredExtensions()
{
    // Ici on parle de instance extensions, pas device extensions
    // GLFW a lui aussi besoin de certaines extensions vulkan pour établir le lien vulkan-fenêtre
    // par exemple GLFW requiert "VK_KHR_surface" pour créer le type abstrait "VkSurfaceKHR"
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    
    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
    
#ifndef NDEBUG
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif
    return extensions;
}

#ifdef VULKAN_RENDERER
vk::SurfaceKHR sk::Window::createSurface(VkInstance instance)
{
    VkSurfaceKHR tmpSurface;
    if(glfwCreateWindowSurface(instance, window, nullptr, &tmpSurface) != VK_SUCCESS)
    {
        throw std::runtime_error("GLFW failed to create window surface !");
    }
    return vk::SurfaceKHR(tmpSurface);
}
#endif