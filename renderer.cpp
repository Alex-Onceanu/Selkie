#define VULKAN_HPP_NO_CONSTRUCTORS
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

#include <iostream>
#include <vector>
#include <string>
#include <exception>
#include <optional>
#include <cstdint>
#include <limits>
#include <fstream>
#include <array>
#include <chrono>

#include "math.hpp"
#include "renderer.hpp"
#include "window.hpp"

// #define RT_WIDTH 3840u
// #define RT_HEIGHT 2160u

#define RT_WIDTH 1920u
#define RT_HEIGHT 1080u

// #define RT_WIDTH 1366u
// #define RT_HEIGHT 768u

// structs
namespace
{
    struct Vertex
    {
        math::vec2 pos;
        math::vec3 clr;
    };

    struct UniformBufferObject {
        alignas(4) float uTime;
        alignas(16) math::vec3 uClr;
    };

    struct QueueFamilyIndices {
        std::optional<uint32_t> graphicsFamily;
        std::optional<uint32_t> presentFamily;
        
        bool isComplete()
        {
            return graphicsFamily.has_value() and presentFamily.has_value();
        }
    };
    
    struct SwapChainSupportDetails {
        vk::SurfaceCapabilitiesKHR capabilities;
        std::vector<vk::SurfaceFormatKHR> formats;
        std::vector<vk::PresentModeKHR> presentModes;
        
        bool isComplete()
        {
            return not formats.empty() and not presentModes.empty();
        }
    };

    struct Buffer {
        vk::Device device{};
        vk::Buffer buf{};
        vk::DeviceMemory memory{};
        vk::DeviceAddress deviceAddress{};

        void destroy()
        {
            device.freeMemory(memory);
            device.destroyBuffer(buf);
        }
    };
};

// attributs
namespace
{
    const int NB_FRAMES_IN_FLIGHT = 2;
    int window_width = 0, window_height = 0;
    std::shared_ptr<sk::Window> window;
    vk::detail::DynamicLoader dl;
    vk::Instance instance;
#ifndef NDEBUG
    vk::DebugUtilsMessengerEXT messenger;
#endif
    vk::SurfaceKHR surface;  // "fenêtre" du point de vue de Vulkan
    vk::Device device;
    vk::PhysicalDevice physicalDevice = VK_NULL_HANDLE;
    vk::Queue graphicsQueue; // handle : interface avec la queue "graphics" de la familyQueue

    // TODO : réorganiser les queues. Avoir une queue qui fait du compute (le raytracing) séparé d'une queue qui fait l'UI
    // on pourrait même séparer encore une queue pour le compute de la physique.
    // btw presentQueue ne sert à rien, ce sera tjrs la même que graphicsQueue sur les pc récents
    vk::Queue presentQueue;  // handle : idem pour present (queue qui s'occupe de donner le rendu à l'écran)

    vk::SwapchainKHR swapChain{}; // file d'images attendant d'être rendues
    vk::Format swapChainImageFormat;
    vk::Extent2D swapChainExtent;
    std::vector<vk::Image> swapChainImages;
    std::vector<vk::ImageView> swapChainImageViews;
    
    vk::CommandPool commandPool;
    std::vector<vk::CommandBuffer> commandBuffers;

    std::vector<vk::DeviceMemory> rtImageMemories;
    std::vector<vk::Image> rtImages;
    std::vector<vk::ImageView> rtImageViews;
    std::vector<vk::DescriptorImageInfo> rtDescImageInfos{};

    Buffer tlasBufs;
    vk::WriteDescriptorSetAccelerationStructureKHR tlasDescInfo;
    vk::AccelerationStructureKHR tlasAccel;
    
    std::vector<Buffer> aabbBufs;
    Buffer tlasInstances;
    Buffer tlasScratchBufs;
    
    Buffer blasScratchBuf;
    vk::AccelerationStructureKHR blasAccel;
    vk::WriteDescriptorSetAccelerationStructureKHR blasDescInfo;
    Buffer blasBuf;
    
    std::vector<vk::StridedDeviceAddressRegionKHR> sbtRegions{};
    std::vector<vk::DescriptorSetLayoutBinding> bindings{};
    std::vector<vk::RayTracingShaderGroupCreateInfoKHR> shaderGroups{};
    std::vector<vk::ShaderModule> shaderModules{};
    std::vector<Buffer> bindingTableBufs{};
    
    vk::DescriptorPool descriptorPool;
    vk::DescriptorSetLayout descriptorSetLayout;
    vk::PipelineLayout pipelineLayout;
    vk::Pipeline pipeline;
    std::vector<vk::DescriptorSet> descriptorSets;

    std::vector<vk::Semaphore> imageAvailableSemaphores;
    std::vector<vk::Fence> readyForNextFrameFences;

    std::vector<vk::Semaphore> readyForPresentationSemaphores; // 1 par swap chain image, pas par frame in flight

    int currentFrame;
    uint32_t currentSwapChainImage;
    bool windowResized;

    const std::vector<const char*> deviceRequiredExtensions = {
        #ifdef __APPLE__
        "VK_KHR_portability_subset",
        #endif
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME
        // TODO : se renseigner sur VK_EXT_ray_tracing_invocation_reorder
    };
};

// methodes
namespace
{
#ifndef NDEBUG
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                        vk::DebugUtilsMessageTypeFlagsEXT messageType,
                                                        const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                                        void* pUserData)
    {
        std::cout << ">> Validation layer says : " << pCallbackData->pMessage << std::endl;
        return VK_FALSE;
    }

    void DestroyDebugUtilsMessengerEXT(vk::Instance instance, vk::DebugUtilsMessengerEXT debugMessenger) {
        auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
        if (func != nullptr) {
            func(instance, debugMessenger, nullptr);
        }
    }
#endif

    void resizeCallback()
    {
        windowResized = true;
    }
    
    void createInstance()
    {
        std::vector<const char*> requiredExtensions = window->getRequiredExtensions();

        vk::ApplicationInfo appInfo{
            .pApplicationName   = "IGR202",
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName        = "Selkie",
            .engineVersion      = VK_MAKE_VERSION(1, 0, 0),
            .apiVersion         = VK_API_VERSION_1_4
        };
        
#if __APPLE__
        requiredExtensions.emplace_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        createInfo.flags |= vk::InstanceCreateFlags(VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR);
#endif

#ifndef NDEBUG
        auto debugInfo = vk::DebugUtilsMessengerCreateInfoEXT()
            .setMessageSeverity(vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo)
            .setMessageType(vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation)
            .setPfnUserCallback(debugCallback);

        std::vector<vk::ValidationFeatureEnableEXT> validationFeatureEnables = { vk::ValidationFeatureEnableEXT::eDebugPrintf };
        auto validationFeatures = vk::ValidationFeaturesEXT()
            .setEnabledValidationFeatures(validationFeatureEnables)
            .setPNext(&debugInfo);

        std::vector<const char*> layers = { "VK_LAYER_KHRONOS_validation"};
        requiredExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif
            
        auto createInfo = vk::InstanceCreateInfo()
            .setPEnabledLayerNames(layers)
            .setPEnabledExtensionNames(requiredExtensions)
            .setPApplicationInfo(&appInfo)
#ifndef NDEBUG
            .setPNext(&validationFeatures);
#else
            ;
#endif
        instance = vk::createInstance(createInfo);
        VULKAN_HPP_DEFAULT_DISPATCHER.init(instance);

#ifndef NDEBUG
        messenger = instance.createDebugUtilsMessengerEXT(debugInfo);
#endif
    }
    
    SwapChainSupportDetails querySwapChainSupport(const vk::PhysicalDevice& gpu)
    {
        return {
            .capabilities = gpu.getSurfaceCapabilitiesKHR(surface),
            .formats = gpu.getSurfaceFormatsKHR(surface),
            .presentModes = gpu.getSurfacePresentModesKHR(surface)
        };
    }
    
    // Une queue family c'est un certain type de queues (elles sont catégorisées selon ce à quoi elles servent)
    QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice device)
    {
        QueueFamilyIndices indices;
        
        std::vector<vk::QueueFamilyProperties> qProperties = device.getQueueFamilyProperties();
        
        int i = 0;
        for(const auto& property : qProperties)
        {
            if(property.queueCount > 0 and property.queueFlags & vk::QueueFlagBits::eCompute)
            {
                indices.graphicsFamily = i;
            }
            
            if(property.queueCount > 0 and device.getSurfaceSupportKHR(i, surface))
            {
                indices.presentFamily = i;
            }
            
            if(indices.isComplete())
            {
                break;
            }
            
            i++;
        }
        
        return indices;
    }
    
    bool deviceSupportsExtensions(vk::PhysicalDevice gpu)
    {
        // s'assure que ce gpu supporte bien les device extensions dont on a besoin
        std::vector<vk::ExtensionProperties> availableExtensions = gpu.enumerateDeviceExtensionProperties();
        
        for(const std::string& extension : deviceRequiredExtensions)
        {
            auto str_eq = [&](vk::ExtensionProperties l){ return extension == l.extensionName.data(); };
            
            if(std::find_if_not(availableExtensions.begin(), availableExtensions.end(), str_eq) == availableExtensions.end())
            {
                // On a trouvé une extension requise non supportée par ce gpu
                return false;
            }
        }
        return true;
    }
    
    vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availableModes)
    {
        for (const auto& availableMode : availableModes)
        {
            if(availableMode == vk::PresentModeKHR::eMailbox)
            {
                return availableMode;
            }
        }

        return vk::PresentModeKHR::eFifo;
    }
    
    vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats)
    {
        for (const auto& availableFormat : availableFormats)
        {
            if(availableFormat.format == vk::Format::eB8G8R8A8Unorm
               and availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
            {
                return availableFormat;
            }
        }
        return availableFormats[0];
    }
    
    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities)
    {
        // Swap extent : resolution des images dans la swap chain
        // capabilities contient les résolutions acceptées
        if(capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
        {
            return capabilities.currentExtent;
        }
        else
        {
            int width, height;
            window->getSize(width, height);
            
            vk::Extent2D actualExtent{
                static_cast<uint32_t>(CLAMP(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width)),
                
                static_cast<uint32_t>(CLAMP(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height))
            };

            return actualExtent;
        }
    }
    
    int rateGPU(vk::PhysicalDevice gpu)
    {
        int score = 0;
        vk::PhysicalDeviceProperties gpuProperties;
        vk::PhysicalDeviceFeatures gpuFeatures;
        gpu.getProperties(&gpuProperties);
        gpu.getFeatures(&gpuFeatures);

        // std::cout << ">> GPU " << gpuProperties.deviceName << "\n";
        
        if(gpuProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu)
        {
            // askip les discrete GPU c'est mieux que tout le reste
            score += 10000;
        }
        
        // Un meilleur GPU supporte des textures plus grandes
        score += gpuProperties.limits.maxImageDimension2D;
        
        QueueFamilyIndices qIndices = findQueueFamilies(gpu);
        if(not qIndices.isComplete())
        {
            // On a besoin que le gpu puisse "comprendre" les commandes
            return 0;
        }
        
        if(not deviceSupportsExtensions(gpu))
        {
            // On a besoin que le gpu supporte toutes les extensions requises
            // std::cout << "does NOT support all extensions :(" << std::endl;
            return 0;
        }
        
        if(not querySwapChainSupport(gpu).isComplete())
        {
            // On a besoin que le gpu supporte le swap chain
            return 0;
        }
        
        // imposer ici que le GPU doive supporter certaines fonctionnalités
        // ex : if(not gpuFeatures.geometryShader) return 0;
        // std::cout << "DOES support all extensions :)" << std::endl;
        return score;
    }
    
    void pickPhysicalDevice()
    {
        std::vector<vk::PhysicalDevice> gpus = instance.enumeratePhysicalDevices();
        if (gpus.size() == 0)
        {
            throw std::runtime_error("No GPU ??");
        }
        
        // On itère sur tous les GPUs disponibles pour choisir "le meilleur" (cf méthode rateGPU)
        int bestScore = 0;
        int tmpScore = 0;
        for(const auto& gpu : gpus)
        {
            tmpScore = rateGPU(gpu);
            if(tmpScore > bestScore)
            {
                bestScore = tmpScore;
                physicalDevice = gpu;
            }
        }
        
        if(bestScore == 0 || physicalDevice == VK_NULL_HANDLE)
        {
            throw std::runtime_error("Found some physical devices, but all are unsuitable.");
        }

        vk::PhysicalDeviceProperties gpuProperties;
        physicalDevice.getProperties(&gpuProperties);
        std::cout << "Using GPU \"" << gpuProperties.deviceName << "\"" << std::endl;
    }
    
    void createLogicalDevice()
    {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        // si on soumet des instructions au GPU en même temps via 2 queues de la même queue family,
        // le gpu va traiter en priorité les commandes de la queue avec la plus grande priorité
        float qPriority = 1.0f; // On va passer a pQueuePriorities un tableau de priorités, ici juste &qPriority
        std::vector<vk::DeviceQueueCreateInfo> allQueuesCreateInfo;
        
        vk::DeviceQueueCreateInfo queueCreateInfo{
            .queueFamilyIndex = indices.graphicsFamily.value(),
            .queueCount = 1,
            .pQueuePriorities = &qPriority
        };
        allQueuesCreateInfo.push_back(queueCreateInfo);
        uint32_t nbNeededQueues = 1;
        
        if(indices.presentFamily.value() != indices.graphicsFamily.value())
        {
            // Pas la peine de push dans allQueuesCreateInfo une 2eme fois la même queue...
            // Si jamais on est pas dans la même queue pour graphics et present, on push la 2eme aussi
            vk::DeviceQueueCreateInfo queueCreateInfo2{
                .queueFamilyIndex = indices.presentFamily.value(),
                .queueCount = 1,
                .pQueuePriorities = &qPriority
            };
            allQueuesCreateInfo.push_back(queueCreateInfo2);
            nbNeededQueues = 2;
        }
        
        vk::PhysicalDeviceFeatures deviceFeatures = vk::PhysicalDeviceFeatures();
        
        // Idem que createInfo pour vk::Instance
        vk::DeviceCreateInfo createInfo{
            .flags = vk::DeviceCreateFlags(),
            .queueCreateInfoCount = nbNeededQueues,
            .pQueueCreateInfos = allQueuesCreateInfo.data(),
            .enabledExtensionCount = static_cast<uint32_t>(deviceRequiredExtensions.size()),
            .ppEnabledExtensionNames = deviceRequiredExtensions.data(),
            .pEnabledFeatures = &deviceFeatures
        };

        auto bufferDeviceAddressFeatures = vk::PhysicalDeviceBufferDeviceAddressFeatures().setBufferDeviceAddress(true);
        auto rayTracingPipelineFeatures = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR().setRayTracingPipeline(true);
        auto accelerationStructureFeatures = vk::PhysicalDeviceAccelerationStructureFeaturesKHR().setAccelerationStructure(true);

        vk::StructureChain createInfoChain{
            createInfo,
            bufferDeviceAddressFeatures,
            rayTracingPipelineFeatures,
            accelerationStructureFeatures
        };

        device = physicalDevice.createDevice(createInfoChain.get<vk::DeviceCreateInfo>());
        VULKAN_HPP_DEFAULT_DISPATCHER.init(device);
        
        graphicsQueue = device.getQueue(indices.graphicsFamily.value(), 0);
        presentQueue = device.getQueue(indices.presentFamily.value(), 0);
    }
    
    void createSurface()
    {
        surface = window->createSurface(instance);
        window->setResizeCallback(resizeCallback);
    }

    void createSwapChain(vk::SwapchainKHR old = VK_NULL_HANDLE)
    {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);
        
        vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        vk::PresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        vk::Extent2D extent = chooseSwapExtent(swapChainSupport.capabilities);
        
        // Combien d'images on laisse dans le swap chain ? 1 + le minimum possible pour que ça fonctionne
        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        const uint32_t maxImgCount = swapChainSupport.capabilities.maxImageCount;
        
        // Si le gpu impose un maximum d'images dans la swap chain (0 := pas de limite) on limite imageCount
        if(maxImgCount > 0 && imageCount > maxImgCount)
        {
            imageCount = maxImgCount;
        }
        
        vk::SwapchainCreateInfoKHR scCreateInfo{
            .flags = vk::SwapchainCreateFlagsKHR(),
            .surface = surface,
            .minImageCount = imageCount,
            .imageFormat = surfaceFormat.format,
            .imageColorSpace = surfaceFormat.colorSpace,
            .imageExtent = extent,
            .imageArrayLayers = 1, // > 1 pour stereoscopic 3D application (VR)
            .imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst,
            
            // on applique pas de transformation finale (rotation, flip..)
            .preTransform = swapChainSupport.capabilities.currentTransform,
            // on pourrait utiliser le canal alpha pour blend avec d'autres fenêtres
            .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
            
            .presentMode = presentMode,
            .clipped = vk::True, // on fait rien avec les pixels cachés par d'autres fenêtre
            .oldSwapchain = old
        };
        
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[2] = {indices.graphicsFamily.value(), indices.presentFamily.value()};
        if(indices.graphicsFamily != indices.presentFamily)
        {
            // Si on utilise 2 queue familles différentes, les deux veulent accéder aux mêmes images
            // dans la swapChain. Elles se le partagent avec le mode concurrent
            scCreateInfo.imageSharingMode = vk::SharingMode::eConcurrent;
            scCreateInfo.queueFamilyIndexCount = 2;
            scCreateInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        else
        {
            // Ce mode de partage transfère l'"ownership" des images. Ici on a une seule famille donc
            // on doit choisir ce mode, car l'autre requiert au moins 2 familles
            scCreateInfo.imageSharingMode = vk::SharingMode::eExclusive;
        }
        
        try {
            swapChain = device.createSwapchainKHR(scCreateInfo);
        } catch (vk::SystemError err) {
            throw std::runtime_error("Failed to create swap chain." + std::string(err.what()));
        }
        
        swapChainImages = device.getSwapchainImagesKHR(swapChain);

        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }

    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties)
    {
        vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) and (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
                return i;
        }

        throw std::runtime_error("Found no compatible memory type.");
    }

    // Fait passer image d'un mode à un autre (lecture / écriture, optimisé pour transfer src / dst, à quelle étape de la pipeline...)
    void setImageLayout(vk::CommandBuffer commandBuffer, vk::Image image, 
        vk::ImageLayout oldLayout, vk::ImageLayout newLayout,
        vk::AccessFlags srcAccess = {}, vk::AccessFlags dstAccess = {},
        vk::PipelineStageFlags srcStage = vk::PipelineStageFlagBits::eAllCommands, 
        vk::PipelineStageFlags dstStage = vk::PipelineStageFlagBits::eAllCommands)
    {
        auto barrier = vk::ImageMemoryBarrier()
            .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
            .setImage(image)
            .setOldLayout(oldLayout)
            .setNewLayout(newLayout)
            .setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1})
            .setSrcAccessMask(srcAccess)
            .setDstAccessMask(dstAccess);
        commandBuffer.pipelineBarrier(srcStage, dstStage, {}, {}, {}, barrier);
    }

    void createRTOutputImages()
    {
        for(int i = 0; i < NB_FRAMES_IN_FLIGHT; i++)
        {
            const auto format = vk::Format::eR8G8B8A8Unorm;
            auto imageInfo = vk::ImageCreateInfo()
                .setImageType(vk::ImageType::e2D)
                .setExtent({RT_WIDTH, RT_HEIGHT, 1})
                .setMipLevels(1)
                .setArrayLayers(1)
                .setFormat(format)
                .setUsage(vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc);
            auto tmpImage = device.createImage(imageInfo);
            
            vk::MemoryRequirements requirements = device.getImageMemoryRequirements(tmpImage);
            uint32_t memoryTypeIndex = findMemoryType(requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
            auto memoryInfo = vk::MemoryAllocateInfo()
                .setAllocationSize(requirements.size)
                .setMemoryTypeIndex(memoryTypeIndex);
            auto tmpMemory = device.allocateMemory(memoryInfo);
            
            device.bindImageMemory(tmpImage, tmpMemory, 0);
            
            auto imageViewInfo = vk::ImageViewCreateInfo()
                .setImage(tmpImage)
                .setViewType(vk::ImageViewType::e2D)
                .setFormat(format)
                .setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});

            auto tmpView = device.createImageView(imageViewInfo);
            
            rtDescImageInfos.push_back(vk::DescriptorImageInfo()
                .setImageView(tmpView)
                .setImageLayout(vk::ImageLayout::eGeneral));
            
            // On a besoin de créer un command buffer ici, pour qu'au moment où le gpu est prêt on fasse passer cet imageView de eUndefined à eGeneral
            auto commandBufferInfo = vk::CommandBufferAllocateInfo()
                .setCommandPool(commandPool)
                .setCommandBufferCount(1);
            
            vk::CommandBuffer commandBuffer = device.allocateCommandBuffers(commandBufferInfo).front();

            commandBuffer.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
            setImageLayout(commandBuffer, tmpImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
            commandBuffer.end();
            
            vk::SubmitInfo submitInfo;
            submitInfo.setCommandBuffers(commandBuffer);
            graphicsQueue.submit(submitInfo);
            graphicsQueue.waitIdle();

            device.freeCommandBuffers(commandPool, commandBuffer);
            
            rtImages.push_back(tmpImage);
            rtImageMemories.push_back(tmpMemory);
            rtImageViews.push_back(tmpView);
        }
    }
    
    void createImageViews()
    {
        for(const auto& image : swapChainImages)
        {
            auto ivCreateInfo = vk::ImageViewCreateInfo()
                .setImage(image)
                .setViewType(vk::ImageViewType::e2D)
                .setFormat(swapChainImageFormat)
                .setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});

            try {
                swapChainImageViews.push_back(device.createImageView(ivCreateInfo));
            }
            catch (vk::SystemError err) {
                throw std::runtime_error("Failed to create image views from swap chain image.");
            }
        }
    }
    
    static std::vector<char> readFile(const std::string& filename)
    {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        
        if(not file.is_open())
        {
            throw std::runtime_error("File not found : " + filename);
        }
        
        // On a ouvert le fichier à la fin pour savoir sa taille
        size_t fileSize = static_cast<size_t>(file.tellg());
        std::vector<char> buffer(fileSize);
        
        // On revient au début du fichier pour lire fileSize octets
        file.seekg(0);
        file.read(buffer.data(), fileSize);
        
        file.close();
        
        return buffer;
    }
    
    void recreateSwapChain()
    {
        device.waitIdle();
        
        window->getSize(window_width, window_height);
        
        while(window_width == 0 or window_height == 0)
        {
            window->getSize(window_width, window_height);
            window->waitForEvent();
        }
        
        for(auto& imageView : swapChainImageViews)
        {
            device.destroyImageView(imageView);
        }
        swapChainImageViews.clear();
        
        auto oldOne = swapChain;
        createSwapChain(oldOne);

        device.destroySwapchainKHR(oldOne);
        createImageViews();
    }

    // Objet vulkan contenant le bytecode d'un shader en SPIR-V
    vk::ShaderModule createShaderModule(const std::vector<char>& code)
    {
        vk::ShaderModuleCreateInfo smCreateInfo{
            .flags = vk::ShaderModuleCreateFlags(),
            .codeSize = code.size(),
            .pCode = reinterpret_cast<const uint32_t*>(code.data()),
        };
        
        try {
            return device.createShaderModule(smCreateInfo);
        } catch (vk::SystemError) {
            throw std::runtime_error("Failed to create shader module.");
        }
    }
    
    void createRaytracingPipeline()
    {
        /* Terminologie :
         * Module := code spriV compilé
         * Stage  := le shader en question (contient un module)
         * Group  := ensemble de stages 
         * - un Group pour rgen, un pour rmiss (sont eGeneral), puis 1 par ~ type d'objet
         * - chaque type d'objet aura son Group contenant un intersection, un closest hit et un any hit
         */

#ifdef _WINDOWS
        auto rgenCode  = readFile("../../../shaders/out/tmp.rgen.spv");
        auto rmissCode = readFile("../../../shaders/out/tmp.rmiss.spv");
        auto rchitSphereCode = readFile("../shaders/out/sphere.rchit.spv");
        auto rintSphereCode  = readFile("../shaders/out/sphere.rint.spv");
        auto rchitGroundCode = readFile("../shaders/out/ground.rchit.spv");
        auto rintGroundCode  = readFile("../shaders/out/ground.rint.spv");
#else
        auto rgenCode  = readFile("../shaders/out/tmp.rgen.spv");
        auto rmissCode = readFile("../shaders/out/tmp.rmiss.spv");
        auto rchitSphereCode = readFile("../shaders/out/sphere.rchit.spv");
        auto rintSphereCode  = readFile("../shaders/out/sphere.rint.spv");
        auto rchitGroundCode = readFile("../shaders/out/ground.rchit.spv");
        auto rintGroundCode  = readFile("../shaders/out/ground.rint.spv");
#endif
        shaderModules.push_back(createShaderModule(rgenCode));
        shaderModules.push_back(createShaderModule(rmissCode));
        shaderModules.push_back(createShaderModule(rchitSphereCode));
        shaderModules.push_back(createShaderModule(rintSphereCode));
        shaderModules.push_back(createShaderModule(rchitGroundCode));
        shaderModules.push_back(createShaderModule(rintGroundCode));

        vk::ShaderStageFlagBits stageTypes[] = { 
            vk::ShaderStageFlagBits::eRaygenKHR,
            vk::ShaderStageFlagBits::eMissKHR,
            vk::ShaderStageFlagBits::eClosestHitKHR,
            vk::ShaderStageFlagBits::eIntersectionKHR,
            vk::ShaderStageFlagBits::eClosestHitKHR,
            vk::ShaderStageFlagBits::eIntersectionKHR
        };

        std::vector<vk::PipelineShaderStageCreateInfo> stages;

        for(int i = 0; i < shaderModules.size(); i++)
        {
            stages.push_back(vk::PipelineShaderStageCreateInfo()
                                .setStage(stageTypes[i])
                                .setModule(shaderModules[i])
                                .setPName("main"));
        }

        // rgen group
        shaderGroups.push_back(vk::RayTracingShaderGroupCreateInfoKHR()
            .setType(vk::RayTracingShaderGroupTypeKHR::eGeneral)
            .setGeneralShader(0)
            .setClosestHitShader(vk::ShaderUnusedKHR)
            .setAnyHitShader(vk::ShaderUnusedKHR)
            .setIntersectionShader(vk::ShaderUnusedKHR));

        // rmiss group
        shaderGroups.push_back(vk::RayTracingShaderGroupCreateInfoKHR()
            .setType(vk::RayTracingShaderGroupTypeKHR::eGeneral)
            .setGeneralShader(1)
            .setClosestHitShader(vk::ShaderUnusedKHR)
            .setAnyHitShader(vk::ShaderUnusedKHR)
            .setIntersectionShader(vk::ShaderUnusedKHR));

        // sphere group
        shaderGroups.push_back(vk::RayTracingShaderGroupCreateInfoKHR()
            .setType(vk::RayTracingShaderGroupTypeKHR::eProceduralHitGroup)
            .setGeneralShader(vk::ShaderUnusedKHR)
            .setClosestHitShader(2)
            .setAnyHitShader(vk::ShaderUnusedKHR)
            .setIntersectionShader(3));

        // Plane group (ground)
        shaderGroups.push_back(vk::RayTracingShaderGroupCreateInfoKHR()
            .setType(vk::RayTracingShaderGroupTypeKHR::eProceduralHitGroup)
            .setGeneralShader(vk::ShaderUnusedKHR)
            .setClosestHitShader(4)
            .setAnyHitShader(vk::ShaderUnusedKHR)
            .setIntersectionShader(5));

        vk::PushConstantRange pushRange;
        pushRange.setOffset(0);
        pushRange.setSize(sizeof(float));
        pushRange.setStageFlags(vk::ShaderStageFlagBits::eRaygenKHR);

        // Ici on envoie aux shaders des valeurs pour les "uniform" (push constants et descriptor sets)
        auto plCreateInfo = vk::PipelineLayoutCreateInfo()
            .setSetLayouts(descriptorSetLayout)
            .setPushConstantRanges(pushRange);
        
        pipelineLayout = device.createPipelineLayout(plCreateInfo);
        
        auto pipelineCreateInfo = vk::RayTracingPipelineCreateInfoKHR()
            .setStages(stages)
            .setGroups(shaderGroups)
            .setMaxPipelineRayRecursionDepth(2)
            .setLayout(pipelineLayout);
        
        auto pipelineCreation = device.createRayTracingPipelineKHR(nullptr, nullptr, pipelineCreateInfo);
        if(pipelineCreation.result != vk::Result::eSuccess)
        {
            throw std::runtime_error("Error when creating ray tracing pipeline !");
        }

        pipeline = pipelineCreation.value;
    }
    
    void createCommandPool()
    {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);
        
        vk::CommandPoolCreateInfo cpCreateInfo {
            .flags = vk::CommandPoolCreateFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer),
            .queueFamilyIndex = queueFamilyIndices.graphicsFamily.value()
        };
        
        try {
            commandPool = device.createCommandPool(cpCreateInfo);
        } catch (vk::SystemError err) {
            throw std::runtime_error("Failed to create command pool.");
        }
    }
    
    // Permet d'écrire les commandes qu'on souhaite dans un command buffer
    // Cette commande s'adresse au rendu sur une image de la swapChain, d'indice imageIndex
    void recordCommandBuffer(vk::CommandBuffer& commandBuffer, uint32_t imageIndex, float t)
    {
        vk::CommandBufferBeginInfo beginInfo {};
        // cf vk::CommandBufferUsageFlagBits::eSimultaneousUse / eRenderPassContinue / eOneTimeSubmit
        
        // Commencer à ajouter des commandes à ce command buffer
        try {
            commandBuffer.begin(beginInfo);
        } catch (vk::SystemError err) {
            throw std::runtime_error("Failed to start recording commands in command buffer.");
        }
        
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, pipeline);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR, pipelineLayout, 0, descriptorSets[currentFrame], nullptr);

        auto instancesData = vk::AccelerationStructureGeometryInstancesDataKHR()
            .setArrayOfPointers(false)
            .setData({ tlasInstances.deviceAddress });

        auto instanceGeometry = vk::AccelerationStructureGeometryKHR()
            .setGeometryType(vk::GeometryTypeKHR::eInstances)
            .setGeometry({ .instances = instancesData })
            .setFlags(vk::GeometryFlagBitsKHR::eOpaque);

        auto buildInfo = vk::AccelerationStructureBuildGeometryInfoKHR()
            .setType(vk::AccelerationStructureTypeKHR::eTopLevel)
            .setFlags(vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate)
            .setMode(vk::BuildAccelerationStructureModeKHR::eUpdate)
            .setSrcAccelerationStructure(tlasAccel)
            .setDstAccelerationStructure(tlasAccel)
            .setGeometries(instanceGeometry)
            .setScratchData({ .deviceAddress = tlasScratchBufs.deviceAddress });

        auto buildRangeInfo = vk::AccelerationStructureBuildRangeInfoKHR()
            .setPrimitiveCount(1)
            .setFirstVertex(0)
            .setPrimitiveOffset(0)
            .setTransformOffset(0);

        commandBuffer.buildAccelerationStructuresKHR(buildInfo, &buildRangeInfo);

        auto barrier = vk::MemoryBarrier()
            .setSrcAccessMask(vk::AccessFlagBits::eAccelerationStructureWriteKHR)
            .setDstAccessMask(vk::AccessFlagBits::eAccelerationStructureReadKHR);

        commandBuffer.pipelineBarrier(
            vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
            vk::PipelineStageFlagBits::eRayTracingShaderKHR,
            {},
            barrier,
            nullptr,
            nullptr
        );

        commandBuffer.pushConstants(pipelineLayout, vk::ShaderStageFlagBits::eRaygenKHR, 0, sizeof(float), &t);
        commandBuffer.traceRaysKHR(sbtRegions[0], sbtRegions[1], sbtRegions[2], {}, RT_WIDTH, RT_HEIGHT, 1u);

        vk::Image srcImage = rtImages[currentFrame];
        vk::Image dstImage = swapChainImages[currentSwapChainImage];

        setImageLayout(commandBuffer, srcImage, vk::ImageLayout::eGeneral, vk::ImageLayout::eTransferSrcOptimal, 
            vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead, 
            vk::PipelineStageFlagBits::eRayTracingShaderKHR, vk::PipelineStageFlagBits::eTransfer);
        setImageLayout(commandBuffer, dstImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, 
            vk::AccessFlagBits::eNone, vk::AccessFlagBits::eTransferWrite, 
            vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer);
        
        auto blitRegion = vk::ImageBlit()
            .setSrcSubresource({ vk::ImageAspectFlagBits::eColor, 0, 0, 1 })
            .setSrcOffsets({ vk::Offset3D{ 0, 0, 0 },
                             vk::Offset3D{static_cast<int32_t>(RT_WIDTH), static_cast<int32_t>(RT_HEIGHT), 1} })
            .setDstSubresource({ vk::ImageAspectFlagBits::eColor, 0, 0, 1 })
            .setDstOffsets({ vk::Offset3D{ 0, 0, 0 },
                             vk::Offset3D{static_cast<int32_t>(window_width), static_cast<int32_t>(window_height), 1} });

        commandBuffer.blitImage(
            srcImage, vk::ImageLayout::eTransferSrcOptimal,
            dstImage, vk::ImageLayout::eTransferDstOptimal,
            blitRegion,
            vk::Filter::eLinear
        );

        // TODO : enlever spécifiquement ce setImageLayout, je crois qu'il est useless
        setImageLayout(commandBuffer, srcImage, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eGeneral,
            vk::AccessFlagBits::eTransferRead, vk::AccessFlagBits::eShaderWrite, 
            vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eRayTracingShaderKHR);

        setImageLayout(commandBuffer, dstImage, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::ePresentSrcKHR,
            vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eNone, 
            vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eBottomOfPipe);
        
        // On a fini d'enregistrer ce qu'on veut que ce commandBuffer fasse
        commandBuffer.end();
    }
    
    void createCommandBuffers()
    {
        commandBuffers.resize(NB_FRAMES_IN_FLIGHT);
        
        vk::CommandBufferAllocateInfo allocInfo {
            .commandPool = commandPool,
            .level = vk::CommandBufferLevel::ePrimary, // secondary : pourrait être appelé depuis des primary
            .commandBufferCount = (uint32_t)commandBuffers.size()
        };
        
        try {
            // allocateCommandBuffers crée ici un std::vector d'un seul élément
            commandBuffers = device.allocateCommandBuffers(allocInfo);
        } catch (vk::SystemError err) {
            throw std::runtime_error("Failed to create command buffer.");
        }
        
        // for(size_t i = 0; i < commandBuffers.size(); i++)
        // {
        //     recordCommandBuffer(commandBuffers[i], (uint32_t)i);
        // }
        
    }
    
    void createSyncObjects()
    {
        // 2 sémaphores : le GPU attend qu'une image de la swap chain ait été trouvée
        // puis que le rendu ait été fini dessus pour la présenter à l'écran
        // 1 fence (= sémaphore pour le CPU) : l'exécution attend la fin de la frame
        // pour pas faire le rendu de 2 frames en même temps
        // et on fait ça pour chaque frame in flight
        
        imageAvailableSemaphores.resize(NB_FRAMES_IN_FLIGHT);
        readyForPresentationSemaphores.resize(swapChainImages.size());
        readyForNextFrameFences.resize(NB_FRAMES_IN_FLIGHT);
        
        vk::FenceCreateInfo fenceInfo {
            // On crée le fence en état "signalé" pour pas que le premier draw call attende indéfiniment
            .flags = vk::FenceCreateFlags(vk::FenceCreateFlagBits::eSignaled)
        };
        
        try {
            for(int i = 0; i < swapChainImages.size(); i++)
            {
                readyForPresentationSemaphores[i] = device.createSemaphore({});
            }
            for(int i = 0; i < NB_FRAMES_IN_FLIGHT; i++)
            {
                imageAvailableSemaphores[i] = device.createSemaphore({});
                readyForNextFrameFences[i] = device.createFence(fenceInfo);
            }
        } catch (vk::SystemError err) {
            throw std::runtime_error("Failed to create semaphores / fence.");
        }
    }
    
    Buffer createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties)
    {
        vk::BufferCreateInfo bufferInfo{
            .size = size,
            .usage = usage,
            .sharingMode = vk::SharingMode::eExclusive  // car utilisé par 1 seule family queue
        };
        
        auto buffer = device.createBuffer(bufferInfo);

        vk::MemoryRequirements memReq = device.getBufferMemoryRequirements(buffer);
    
        vk::MemoryAllocateInfo allocInfo{
            .allocationSize = memReq.size,
            .memoryTypeIndex = findMemoryType(memReq.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)
        };

        auto flagsInfo = vk::MemoryAllocateFlagsInfo().setFlags(vk::MemoryAllocateFlagBits::eDeviceAddress);
        if(usage & vk::BufferUsageFlagBits::eShaderDeviceAddress)
        {
            allocInfo.setPNext(&flagsInfo);
        }
        
        auto bufferMemory = device.allocateMemory(allocInfo);

        device.bindBufferMemory(buffer, bufferMemory, 0);

        vk::BufferDeviceAddressInfoKHR bdaInfo{
            .sType = vk::StructureType::eBufferDeviceAddressInfo,
            .buffer = buffer
        };
        auto bufferAddress = device.getBufferAddressKHR(&bdaInfo);

        return Buffer({.device = device, .buf = buffer, .memory = bufferMemory, .deviceAddress = bufferAddress});
    }
    
    void copyBuffer(vk::Buffer& srcBuf, vk::Buffer& dstBuf, vk::DeviceSize size)
    {
        vk::CommandBufferAllocateInfo allocInfo {
            .commandPool = commandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1
        };
        
        vk::CommandBuffer commandBuffer = device.allocateCommandBuffers(allocInfo).front();
        
        commandBuffer.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        
        vk::BufferCopy copyRegion {
            .size = size
        };
        
        commandBuffer.copyBuffer(srcBuf, dstBuf, 1, &copyRegion);
        commandBuffer.end();
        
        vk::SubmitInfo submitInfo {
            .commandBufferCount = 1,
            .pCommandBuffers = &commandBuffer,
        };
        
        graphicsQueue.submit(submitInfo);
        graphicsQueue.waitIdle();

        device.freeCommandBuffers(commandPool, commandBuffer);
    }
    
    void createAccelerationStructures()
    {
        // BLAS d'abord
        // Spheres, puis plans, etc (plusieurs types de géométries pour le même BLAS, utiliseront un hitgroup différent)
        std::vector<std::vector<vk::AabbPositionsKHR>> aabbs = {{{-1.0f, 0.0f, -1.0f, 1.0f, 20.0f, 1.0f}}, {{-1000.f, -0.1f, -1000.f, 1000.f, 0.1f, 1000.f}}};

        // le blas ne peut être construit qu'une fois que copyBuffer est fini, il faut une barrière
        std::vector<vk::BufferMemoryBarrier> barriers{};
        std::vector<uint32_t> primitiveCounts{};
        for(const auto& aabb : aabbs)
        {
            vk::DeviceSize bufSize = aabb.size() * sizeof(aabb[0]);

            auto stagingBuf = createBuffer(bufSize, vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eShaderDeviceAddress,
                        vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostVisible);
    
            void* data = device.mapMemory(stagingBuf.memory, 0, bufSize);
            memcpy(data, aabb.data(), (size_t)bufSize);
            device.unmapMemory(stagingBuf.memory);
            
            // on met le staging buffer dans le vrai buffer
            aabbBufs.push_back(createBuffer(bufSize,  vk::BufferUsageFlagBits::eShaderDeviceAddress 
                                                    | vk::BufferUsageFlagBits::eTransferDst 
                                                    | vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR, 
                                             vk::MemoryPropertyFlagBits::eDeviceLocal));

            copyBuffer(stagingBuf.buf, aabbBufs.back().buf, bufSize);
            stagingBuf.destroy();

            barriers.push_back(vk::BufferMemoryBarrier()
                .setSrcAccessMask(vk::AccessFlagBits::eTransferWrite)
                .setDstAccessMask(vk::AccessFlagBits::eAccelerationStructureReadKHR)
                .setBuffer       (aabbBufs.back().buf)
                .setSize         (VK_WHOLE_SIZE));

            primitiveCounts.push_back(aabb.size());
        }

        auto tmpCommandBufInfo = vk::CommandBufferAllocateInfo()
            .setCommandPool(commandPool)
            .setCommandBufferCount(1);

        vk::CommandBuffer tmpCommandBuf = device.allocateCommandBuffers(tmpCommandBufInfo).front();
        tmpCommandBuf.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        tmpCommandBuf.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
            {}, 0, nullptr, 2, barriers.data(), 0, nullptr
        );
        tmpCommandBuf.end();

        auto tmpSubmitInfo = vk::SubmitInfo()
            .setCommandBuffers(tmpCommandBuf);
        graphicsQueue.submit({tmpSubmitInfo});
        graphicsQueue.waitIdle();

        std::vector<vk::AccelerationStructureGeometryKHR> geometries{};
        for(const auto& aabbBuf : aabbBufs)
        {
            auto tmpAabbData = vk::AccelerationStructureGeometryAabbsDataKHR()
                .setData({aabbBuf.deviceAddress})
                .setStride(sizeof(vk::AabbPositionsKHR));

            geometries.push_back(vk::AccelerationStructureGeometryKHR()
                .setGeometryType(vk::GeometryTypeKHR::eAabbs)
                .setGeometry({.aabbs = tmpAabbData})
                .setFlags(vk::GeometryFlagBitsKHR::eOpaque)); // TODO : remove this
        }

        // remplir blasBuf, blasBufMemory, blasBufDeviceAddress, blasAccel, blasDescInfo
        auto buildGeometryInfo = vk::AccelerationStructureBuildGeometryInfoKHR()
            .setType(vk::AccelerationStructureTypeKHR::eBottomLevel)
            .setFlags(vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate)
            .setGeometries(geometries);

        vk::AccelerationStructureBuildSizesInfoKHR buildSizesInfo = device.getAccelerationStructureBuildSizesKHR(
            vk::AccelerationStructureBuildTypeKHR::eDevice, buildGeometryInfo, primitiveCounts); 
                
        vk::DeviceSize size = buildSizesInfo.accelerationStructureSize;
        blasBuf = createBuffer(size, vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR 
                                    | vk::BufferUsageFlagBits::eShaderDeviceAddress, 
                                        vk::MemoryPropertyFlagBits::eDeviceLocal); 
        
        auto accelInfo = vk::AccelerationStructureCreateInfoKHR()
            .setBuffer(blasBuf.buf)
            .setSize(size)
            .setType(vk::AccelerationStructureTypeKHR::eBottomLevel);
            
        blasAccel = device.createAccelerationStructureKHR(accelInfo);
            
        blasScratchBuf = createBuffer(buildSizesInfo.buildScratchSize, 
                                                vk::BufferUsageFlagBits::eStorageBuffer 
                                            | vk::BufferUsageFlagBits::eShaderDeviceAddress,
                                                vk::MemoryPropertyFlagBits::eDeviceLocal);

        buildGeometryInfo.setScratchData({blasScratchBuf.deviceAddress})
            .setDstAccelerationStructure(blasAccel);
                        
        // TODO : est-ce que c'est vrmt compatible avec notre command pool ?
        // on alloue un nouveau command buf qui servira à construire le blas
        // donc sera submit UNE seule fois au lancement du programme
        auto commandBufferInfo = vk::CommandBufferAllocateInfo()
            .setCommandPool(commandPool)
            .setCommandBufferCount(1);
            
        vk::CommandBuffer blasCommandBuffer = device.allocateCommandBuffers(commandBufferInfo).front();
            
        // on record la construction du blas
        blasCommandBuffer.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        auto buildRangeInfo = vk::AccelerationStructureBuildRangeInfoKHR()
            .setPrimitiveCount(2)
            .setFirstVertex(0)
            .setPrimitiveOffset(0)
            .setTransformOffset(0);
        blasCommandBuffer.buildAccelerationStructuresKHR(buildGeometryInfo, &buildRangeInfo);
        blasCommandBuffer.end();
        
        auto blasSubmitInfo = vk::SubmitInfo()
            .setCommandBuffers(blasCommandBuffer);
        graphicsQueue.submit({blasSubmitInfo});
        graphicsQueue.waitIdle();

        blasDescInfo.setAccelerationStructures(blasAccel);
        
        // TLAS time babyy
        vk::TransformMatrixKHR transformMatrix{};
        transformMatrix.setMatrix(std::array{
            std::array{1.0f, 0.0f, 0.0f, 0.0f},
            std::array{0.0f, 1.0f, 0.0f, 0.0f},
            std::array{0.0f, 0.0f, 1.0f, 0.0f},
            });

        auto accelInstance = vk::AccelerationStructureInstanceKHR()
            .setTransform(transformMatrix)
            .setMask(0xFF)
            .setAccelerationStructureReference(blasBuf.deviceAddress)
            .setFlags(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable);
            
        tlasInstances = createBuffer(sizeof(vk::AccelerationStructureInstanceKHR), 
                                            vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR 
                                          | vk::BufferUsageFlagBits::eStorageBuffer 
                                          | vk::BufferUsageFlagBits::eShaderDeviceAddress,
                                            vk::MemoryPropertyFlagBits::eHostCoherent 
                                          | vk::MemoryPropertyFlagBits::eHostVisible);
        
        {
            void* data = device.mapMemory(tlasInstances.memory, 0, sizeof(vk::AccelerationStructureInstanceKHR));
            memcpy(data, &accelInstance, sizeof(vk::AccelerationStructureInstanceKHR));
            device.unmapMemory(tlasInstances.memory);
        }

        auto instancesData = vk::AccelerationStructureGeometryInstancesDataKHR()
            .setArrayOfPointers(false)
            .setData({tlasInstances.deviceAddress});
            
        auto instanceGeometry = vk::AccelerationStructureGeometryKHR()
            .setGeometryType(vk::GeometryTypeKHR::eInstances)
            .setGeometry({.instances = instancesData})
            .setFlags(vk::GeometryFlagBitsKHR::eOpaque);
            
        // remplir tlasBuf, tlasBufMemory, tlasBufDeviceAddress, tlasAccel, tlasDescInfo
        auto tlasBuildGeometryInfo = vk::AccelerationStructureBuildGeometryInfoKHR()
            .setType(vk::AccelerationStructureTypeKHR::eTopLevel)
            .setFlags(vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate)
            .setMode(vk::BuildAccelerationStructureModeKHR::eBuild)
            .setGeometries(instanceGeometry);
            
        vk::AccelerationStructureBuildSizesInfoKHR tlasBuildSizesInfo = device.getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice, tlasBuildGeometryInfo, 1);
        vk::DeviceSize tlasSize = tlasBuildSizesInfo.accelerationStructureSize;
        tlasBufs = createBuffer(tlasSize, 
                                        vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR 
                                        | vk::BufferUsageFlagBits::eShaderDeviceAddress, 
                                        vk::MemoryPropertyFlagBits::eDeviceLocal); 
            
        auto tlasAccelInfo = vk::AccelerationStructureCreateInfoKHR()
            .setBuffer(tlasBufs.buf)
            .setSize(tlasSize)
            .setType(vk::AccelerationStructureTypeKHR::eTopLevel);
        tlasAccel = device.createAccelerationStructureKHR(tlasAccelInfo);
            
        tlasScratchBufs = createBuffer(tlasBuildSizesInfo.buildScratchSize, 
                                                vk::BufferUsageFlagBits::eStorageBuffer 
                                            | vk::BufferUsageFlagBits::eShaderDeviceAddress,
                                                vk::MemoryPropertyFlagBits::eDeviceLocal);
                
        tlasBuildGeometryInfo.setScratchData({tlasScratchBufs.deviceAddress});
        tlasBuildGeometryInfo.setDstAccelerationStructure(tlasAccel);
                
        // idem que pour la commande de construction du BLAS
        auto tlasCommandBufferInfo = vk::CommandBufferAllocateInfo()
            .setCommandPool(commandPool)
            .setCommandBufferCount(1);
            
        vk::CommandBuffer tlasCommandBuffer = device.allocateCommandBuffers(tlasCommandBufferInfo).front();

        tlasCommandBuffer.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        auto tlasBuildRangeInfo = vk::AccelerationStructureBuildRangeInfoKHR()
            .setPrimitiveCount(1) // 1 seul noeud dans notre TLAS pour l'instant (y'a 1 objet)
            .setFirstVertex(0)
            .setPrimitiveOffset(0)
            .setTransformOffset(0);
        tlasCommandBuffer.buildAccelerationStructuresKHR(tlasBuildGeometryInfo, &tlasBuildRangeInfo);
        tlasCommandBuffer.end();
            
        auto tlasSubmitInfo = vk::SubmitInfo()
            .setCommandBuffers(tlasCommandBuffer);
        graphicsQueue.submit({tlasSubmitInfo});
        graphicsQueue.waitIdle();

        device.freeCommandBuffers(commandPool, tlasCommandBuffer);
            
        tlasDescInfo.setAccelerationStructures(tlasAccel);
    }

    void createDescriptorSetLayout()
    {
        bindings.push_back({0, vk::DescriptorType::eAccelerationStructureKHR, 1, vk::ShaderStageFlagBits::eRaygenKHR});   // Binding = 0 : TLAS
        bindings.push_back({1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eRaygenKHR});               // Binding = 1 : Storage image
        // {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eClosestHitKHR},                        // Binding = 2 : Vertices
        // {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eClosestHitKHR}                         // Binding = 3 : Indices
        
        auto layoutInfo = vk::DescriptorSetLayoutCreateInfo()
            .setBindings(bindings);

        descriptorSetLayout = device.createDescriptorSetLayout(layoutInfo);
    }
    
    void createDescriptorPool()
    {
        std::vector<vk::DescriptorPoolSize> poolSizes{
            {vk::DescriptorType::eAccelerationStructureKHR, NB_FRAMES_IN_FLIGHT},
            {vk::DescriptorType::eStorageImage, NB_FRAMES_IN_FLIGHT}
            // {vk::DescriptorType::eStorageBuffer, 0 * NB_FRAMES_IN_FLIGHT}, // en vrai il en faut ((vertexbuf + indexbuf) * nb_meshes + materialsbuf) * NB_FRAMES_IN_FLIGHT
        };

        vk::DescriptorPoolCreateInfo poolInfo;
        poolInfo.setPoolSizes(poolSizes);
        poolInfo.setMaxSets(NB_FRAMES_IN_FLIGHT);
        poolInfo.setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);
        
        descriptorPool = device.createDescriptorPool(poolInfo);
    }
    
    void createDescriptorSets()
    {
        // Avant de créer les descriptor sets on crée la Shader Binding Table 
        // cher gpu, avec quel alignement mémoire dois-je construire ma shader binding table ?
        auto properties = physicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
        auto rtProperties = properties.get<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();

        uint32_t handleSize = rtProperties.shaderGroupHandleSize;
        uint32_t handleSizeAligned = rtProperties.shaderGroupHandleAlignment;
        // on arrondit handleSize au multiple de handleSizeAligned le plus proche (par excès)
        uint32_t stride = (handleSize + handleSizeAligned - 1) & ~(handleSizeAligned - 1); // evil bit hack
        uint32_t groupCount = static_cast<uint32_t>(shaderGroups.size());
        uint32_t sbtSize = groupCount * handleSize;

        // TODO : se renseigner sur shader record (comment utiliser des matériaux (ou autres params) différents par instance ?)

        // globalement ici on demande l'agencement mémoire des shaders qu'on a chargés, pour les donner aux shaders (via les descriptor sets)
        // mais pk des shaders auraient besoin de connaitre l'adresse mémoire de leur code source ? Astuce de Quine ??
        // en fait c'est parce qu'ils vont pouvoir s'appeler les uns les autres récursviement (ex : chit appelle raygen pour ombres / réflections)
        std::vector<uint8_t> handleStorage(sbtSize);
        if (device.getRayTracingShaderGroupHandlesKHR(pipeline, 0, groupCount, sbtSize, handleStorage.data()) != vk::Result::eSuccess) {
            throw std::runtime_error("Error when getting raytracing shader group handles.");
        }

        {
            auto rgenTable = createBuffer(stride, vk::BufferUsageFlagBits::eShaderBindingTableKHR 
                                                    | vk::BufferUsageFlagBits::eShaderDeviceAddress,
                                        vk::MemoryPropertyFlagBits::eDeviceLocal);
            
            // on écrit une adresse mémoire à une adresse mémoire
            // par contre ce qui m'étonne c'est qu'on écrit l'adresse mémoire de son propre programme au lieu d'écrire celle des autres shaders
            void* mapped = device.mapMemory(rgenTable.memory, 0, stride);
            memcpy(mapped, handleStorage.data() + 0 * handleSize, handleSize);
            device.unmapMemory(rgenTable.memory);

            bindingTableBufs.push_back(rgenTable);
        }

        {
            auto rmissTable = createBuffer(stride, vk::BufferUsageFlagBits::eShaderBindingTableKHR 
                                                    | vk::BufferUsageFlagBits::eShaderDeviceAddress,
                                        vk::MemoryPropertyFlagBits::eDeviceLocal);
            
            void* mapped = device.mapMemory(rmissTable.memory, 0, stride);
            memcpy(mapped, handleStorage.data() + 1 * handleSize, handleSize);
            device.unmapMemory(rmissTable.memory);

            bindingTableBufs.push_back(rmissTable);
        }
        
        {
            auto rhitTable = createBuffer(2 * stride, vk::BufferUsageFlagBits::eShaderBindingTableKHR 
                                                        | vk::BufferUsageFlagBits::eShaderDeviceAddress,
                                          vk::MemoryPropertyFlagBits::eDeviceLocal);
            
            uint8_t* mappedSph = (uint8_t*)device.mapMemory(rhitTable.memory, 0, 2 * stride);
            memcpy(mappedSph, handleStorage.data() + 2 * handleSize, handleSize);
            memcpy(mappedSph + stride, handleStorage.data() + 3 * handleSize, handleSize);
            device.unmapMemory(rhitTable.memory);

            bindingTableBufs.push_back(rhitTable);
        }

        sbtRegions.push_back({bindingTableBufs[0].deviceAddress, stride, handleSize});
        sbtRegions.push_back({bindingTableBufs[1].deviceAddress, stride, handleSize});
        sbtRegions.push_back({bindingTableBufs[2].deviceAddress, stride, 2 * handleSize});

        std::vector<vk::DescriptorSetLayout> layouts(NB_FRAMES_IN_FLIGHT, descriptorSetLayout);
        auto descSetInfo = vk::DescriptorSetAllocateInfo()
            .setDescriptorPool(descriptorPool)
            .setSetLayouts(layouts);
        
        descriptorSets = device.allocateDescriptorSets(descSetInfo);
        
        for(int frame = 0; frame < NB_FRAMES_IN_FLIGHT; frame++)
        {
            std::vector<vk::WriteDescriptorSet> writes(bindings.size());
            for (int i = 0; i < bindings.size(); i++) {
                writes[i].setDstSet(descriptorSets[frame]);
                writes[i].setDescriptorType(bindings[i].descriptorType);
                writes[i].setDescriptorCount(bindings[i].descriptorCount);
                writes[i].setDstBinding(bindings[i].binding);
            }
            writes[0].setPNext(tlasDescInfo);
            writes[1].setImageInfo(rtDescImageInfos[frame]);
            device.updateDescriptorSets(writes, nullptr);
        }
    }

    void initVulkan()
    {
        
        PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
        VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);
        std::vector<vk::ExtensionProperties> extensions = vk::enumerateInstanceExtensionProperties(nullptr);

        createInstance();

        createSurface();
        
        // Choisir le GPU
        pickPhysicalDevice();
        
        // Instancier l'interface avec le GPU
        createLogicalDevice();
        
        // File d'images qui attendent d'être affichées
        createSwapChain();
        
        // Pour chaque image de la swap chain, créer son imageView (comment l'interpréter)
        createImageViews();
        
        // Description de comment associer des uniform GLSL à chaque morceau de l'Uniform Buffer Object
        // En fait décrit à quoi va ressembler le descriptor set qui lui contiendra les uniform
        // D'ailleurs pas que les uniform ! Tout ce qui est attributs et autres inputs de chaque shader
        createDescriptorSetLayout();
        
        // On charge les shaders compilés en .spv et on les lie
        createRaytracingPipeline();
        
        // Tableau de command buffers. Mais on en a un seul
        createCommandPool();

        // Maintenant on crée les images qui serviront d'output au raygen shader (1 par frame in flight)
        createRTOutputImages();

        // Arbre qui contiendra tous nos objets (TLAS), + pour chaque objet un arbre qui stocke ses primitives (BLAS)
        createAccelerationStructures();
        
        // Équivalent de commandPool mais pour uniform buffer
        createDescriptorPool();
        
        // Ce qu'on envoie au GPU (contient les uniform et est décrit par son descriptorSetLayout)
        createDescriptorSets();
        
        // Enregistrement des commandes qu'on veut faire pour le draw call
        createCommandBuffers();
        
        // Créer les 2 sémaphores et le fence
        createSyncObjects();
    }
};

void sk::end()
{
    device.waitIdle();

    for(int i = 0; i < swapChainImages.size(); i++)
    {
        device.destroySemaphore(readyForPresentationSemaphores[i]);
    }

    for(int f = 0; f < NB_FRAMES_IN_FLIGHT; f++)
    {
        device.destroyFence(readyForNextFrameFences[f]);
        device.destroySemaphore(imageAvailableSemaphores[f]);
        device.destroyImageView(rtImageViews[f]);
        device.freeMemory(rtImageMemories[f]);
        device.destroyImage(rtImages[f]);
    }
    blasBuf.destroy();
    tlasBufs.destroy();
    device.destroyAccelerationStructureKHR(blasAccel);
    device.destroyAccelerationStructureKHR(tlasAccel);
    for(auto& a : aabbBufs) a.destroy();
    tlasInstances.destroy();
    tlasScratchBufs.destroy();
    blasScratchBuf.destroy();
    
    device.destroyPipeline(pipeline);
    device.destroyPipelineLayout(pipelineLayout);
    device.destroyDescriptorSetLayout(descriptorSetLayout);
    device.destroyDescriptorPool(descriptorPool);
    
    for(int i = 0; i < bindingTableBufs.size(); i++)
    {
        bindingTableBufs[i].destroy();
    }
    for(int i = 0; i < shaderModules.size(); i++)
    {
        device.destroyShaderModule(shaderModules[i]);
    }

    device.freeCommandBuffers(commandPool, commandBuffers);
    device.destroyCommandPool(commandPool);
    
    for(auto& imageView : swapChainImageViews)
    {
        device.destroyImageView(imageView);
    }
    device.destroySwapchainKHR(swapChain);

    device.destroy();
    
#ifndef NDEBUG
    DestroyDebugUtilsMessengerEXT(instance, messenger);
#endif
    instance.destroySurfaceKHR(surface);
    instance.destroy();
}

std::shared_ptr<sk::Window> sk::initWindow(unsigned int width, unsigned int height)
{
    currentFrame = 0;
    windowResized = false;
    window_width = width, window_height = height;
    window = std::make_shared<sk::Window>(width, height);

    initVulkan();

    return window;
}

void sk::draw(float t)
{
    // On commence par attendre que la frame précédente soit finie
    if(device.waitForFences(1, &readyForNextFrameFences[currentFrame], vk::True, UINT64_MAX) != vk::Result::eSuccess)
    {
        throw std::runtime_error("Timeout or error during waitForFences !");
    }
    
    // Obtient la prochaine image dispo de la swap chain, et puis fait un post dans imageAvailable
    try {
        currentSwapChainImage = device.acquireNextImageKHR(swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], nullptr).value;
    } catch (vk::OutOfDateKHRError(const std::string &msg)) {
        recreateSwapChain();
        return;
    }
    
    // On reset le fence ment si on doit pas recréer la swap chain (évite une famine)
    device.resetFences(readyForNextFrameFences[currentFrame]);
    
    // Ensuite il faut record ce qu'on veut faire dans commandBuffer, pour l'image d'indice imgId
    // commandBuffers[currentFrame].reset();
    recordCommandBuffer(commandBuffers[currentFrame], currentSwapChainImage, t);
    
    // On voudra attendre le sémaphore imageAvailable au moment de la copie de rtImage sur swapChainImage
    vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eTransfer };
    
    // Ensuite on peut submit le command buffer
    vk::SubmitInfo submitInfo {
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &imageAvailableSemaphores[currentFrame],
        .pWaitDstStageMask = waitStages,
        .commandBufferCount = 1,
        .pCommandBuffers = &commandBuffers[currentFrame],
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &readyForPresentationSemaphores[currentSwapChainImage]
    };

    graphicsQueue.submit(submitInfo, readyForNextFrameFences[currentFrame]);
    
    // Reste plus qu'à envoyer le résultat du rendu à la swap chain pour qu'on puisse le voir
    vk::PresentInfoKHR presentInfo {
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &readyForPresentationSemaphores[currentSwapChainImage],
        .swapchainCount = 1,
        .pSwapchains = &swapChain,
        .pImageIndices = &currentSwapChainImage,
        .pResults = nullptr
    };
    
    vk::Result presentRes;
    try {
        presentRes = presentQueue.presentKHR(presentInfo);
    } catch (vk::OutOfDateKHRError(const std::string &msg)) {
        recreateSwapChain();
        return;
    }
    
    if(presentRes == vk::Result::eSuboptimalKHR or windowResized)
    {
        windowResized = false;
        recreateSwapChain();
    }

    currentFrame = (1 + currentFrame) % NB_FRAMES_IN_FLIGHT;
}
