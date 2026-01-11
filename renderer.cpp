#define VULKAN_HPP_NO_CONSTRUCTORS
#include <vulkan/vulkan.hpp>

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

#define RT_WIDTH 1366u
#define RT_HEIGHT 768u

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
};

// attributs
namespace
{
    std::shared_ptr<sk::Window> window;
    int window_width = 0, window_height = 0;
    vk::detail::DynamicLoader dl;
    vk::UniqueInstance instance;
    vk::PhysicalDevice physicalDevice = VK_NULL_HANDLE;
    vk::UniqueDevice logicalDevice;
    vk::Queue graphicsQueue; // handle : interface avec la queue "graphics" de la familyQueue

    // TODO : réorganiser les queues. Avoir une queue qui fait du compute (le raytracing) séparé d'une queue qui fait l'UI
    // on pourrait même séparer encore une queue pour le compute de la physique.
    // btw presentQueue ne sert à rien, ce sera tjrs la même que graphicsQueue sur les pc récents
    vk::Queue presentQueue;  // handle : idem pour present (queue qui s'occupe de donner le rendu à l'écran)

    vk::UniqueSurfaceKHR surface;  // "fenêtre" du point de vue de Vulkan
    vk::UniqueSwapchainKHR swapChain{}; // file d'images attendant d'être rendues
    vk::Format swapChainImageFormat;
    vk::Extent2D swapChainExtent;
    std::vector<vk::Image> swapChainImages; // et non pas <vk::UniqueImage> pcq c'est l'UniqueSwapchain qui a leur ownership
    std::vector<vk::UniqueImageView> swapChainImageViews;
    std::vector<vk::UniqueImage> rtImages;
    std::vector<vk::UniqueDeviceMemory> rtImageMemories;
    std::vector<vk::UniqueImageView> rtImageViews;
    std::vector<vk::DescriptorImageInfo> rtDescImageInfos{};
    vk::UniqueDescriptorSetLayout descriptorSetLayout; // description de comment lier l'UBO du CPU avec celui du GPU
    vk::UniquePipelineLayout pipelineLayout; // envoi d'uniform dans les shaders
    vk::UniquePipeline pipeline;
    vk::UniqueCommandPool commandPool;
    vk::UniqueDescriptorPool descriptorPool;
    std::vector<vk::UniqueDescriptorSet> descriptorSets;
    int currentFrame;
    bool windowResized;

    const std::vector<Vertex> vertices {
        { { -1.,-1. },{ 1.,0.,0. } },
        { { -1., 1. },{ 0.,1.,0. } },
        { {  1., 1. },{ 0.,0.,1. } },
        { {  1.,-1. },{ 1.,0.,1. } }
    };
    
    const std::vector<uint16_t> indices {
        0, 1, 3,
        3, 1, 2
    };
    
    // pour chaque frame in flight
    std::vector<vk::CommandBuffer, std::allocator<vk::CommandBuffer>> commandBuffers;
    std::vector<vk::UniqueSemaphore> imageAvailableSemaphores;
    std::vector<vk::UniqueSemaphore> readyForPresentationSemaphores;
    std::vector<vk::UniqueFence> readyForNextFrameFences;
    
    vk::UniqueBuffer vertexBuffer;
    vk::UniqueDeviceMemory vertexBufferMemory;
    vk::DeviceAddress vertexBufferDeviceAddress;
    vk::UniqueBuffer indexBuffer;
    vk::UniqueDeviceMemory indexBufferMemory;
    vk::DeviceAddress indexBufferDeviceAddress;
    
    std::vector<vk::UniqueBuffer> uniformBuffers;
    std::vector<vk::UniqueDeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;

    vk::UniqueBuffer tlasBuf;
    vk::UniqueDeviceMemory tlasBufMemory;
    vk::DeviceAddress tlasBufDeviceAddress;
    vk::UniqueAccelerationStructureKHR tlas;
    vk::WriteDescriptorSetAccelerationStructureKHR tlasDescInfo;
    vk::UniqueAccelerationStructureKHR tlasAccel;

    vk::UniqueBuffer blasBuf;
    vk::UniqueDeviceMemory blasBufMemory;
    vk::DeviceAddress blasBufDeviceAddress;
    vk::UniqueAccelerationStructureKHR blas;
    vk::WriteDescriptorSetAccelerationStructureKHR blasDescInfo;
    vk::UniqueAccelerationStructureKHR blasAccel;

    std::vector<vk::RayTracingShaderGroupCreateInfoKHR> shaderGroups{};
    std::vector<vk::DescriptorSetLayoutBinding> bindings{};
    std::vector<vk::StridedDeviceAddressRegionKHR> regions{};
    
    const std::vector<const char*> deviceRequiredExtensions = {
#ifdef __APPLE__
        "VK_KHR_portability_subset",
#endif
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME
        // TODO : se renseigner sur VK_EXT_ray_tracing_invocation_reorder
    };
    
    const int NB_FRAMES_IN_FLIGHT = 2;
};

// methodes
namespace
{
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
        
        vk::InstanceCreateInfo createInfo{ .pApplicationInfo = &appInfo };

        createInfo.enabledLayerCount = 0;
        createInfo.pNext = nullptr;

#if __APPLE__
        requiredExtensions.emplace_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        createInfo.flags |= vk::InstanceCreateFlags(VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR);
#endif
        
        createInfo.enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size());
        createInfo.ppEnabledExtensionNames = requiredExtensions.data();
        
        instance = vk::createInstanceUnique(createInfo);
    }
    
    SwapChainSupportDetails querySwapChainSupport(const vk::PhysicalDevice& gpu)
    {
        return {
            .capabilities = gpu.getSurfaceCapabilitiesKHR(surface.get()),
            .formats = gpu.getSurfaceFormatsKHR(surface.get()),
            .presentModes = gpu.getSurfacePresentModesKHR(surface.get())
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
            
            if(property.queueCount > 0 and device.getSurfaceSupportKHR(i, surface.get()))
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
            auto str_eq = [&](vk::ExtensionProperties l){ return extension.compare(std::string(l.extensionName)); };
            
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

        // std::cout << "GPU " << gpuProperties.deviceName << " : ";
        
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
        std::vector<vk::PhysicalDevice> gpus = instance->enumeratePhysicalDevices();
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

        vk::PhysicalDeviceBufferDeviceAddressFeatures bufferDeviceAddressFeatures{vk::StructureType::ePhysicalDeviceBufferDeviceAddressFeaturesKHR};
        vk::PhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipelineFeatures{vk::StructureType::ePhysicalDeviceRayTracingPipelineFeaturesKHR};
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures{vk::StructureType::ePhysicalDeviceAccelerationStructureFeaturesKHR};
        vk::StructureChain createInfoChain{
            createInfo,
            bufferDeviceAddressFeatures,
            rayTracingPipelineFeatures,
            accelerationStructureFeatures
        };

        logicalDevice = physicalDevice.createDeviceUnique(createInfoChain.get<vk::DeviceCreateInfo>());
        
        graphicsQueue = logicalDevice->getQueue(indices.graphicsFamily.value(), 0);
        presentQueue = logicalDevice->getQueue(indices.presentFamily.value(), 0);
    }
    
    void createSurface()
    {
        surface = window->createSurface(instance.get());
        window->setResizeCallback(resizeCallback);
    }
    
    void cleanupSwapChain()
    {
        swapChainImageViews.clear();
    }

    void createSwapChain()
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
            .surface = surface.get(),
            .minImageCount = imageCount,
            .imageFormat = surfaceFormat.format,
            .imageColorSpace = surfaceFormat.colorSpace,
            .imageExtent = extent,
            .imageArrayLayers = 1, // > 1 pour stereoscopic 3D application (VR)
            .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
            
            // on applique pas de transformation finale (rotation, flip..)
            .preTransform = swapChainSupport.capabilities.currentTransform,
            // on pourrait utiliser le canal alpha pour blend avec d'autres fenêtres
            .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
            
            .presentMode = presentMode,
            .clipped = vk::True, // on fait rien avec les pixels cachés par d'autres fenêtre
            .oldSwapchain = swapChain ? swapChain.get() : vk::SwapchainKHR(VK_NULL_HANDLE) // pour les fenêtres redimensionnables, recréer une swap chain.
        };
        
        /*  dans imageUsage on dit qu'on prend les images de la swap chain pour directement
            les envoyer au rendu. On pourrait plutôt les envoyer ailleurs (ex : pour faire du post-processing?)
            avec VK_IMAGE_USAGE_TRANSFER_DST_BIT */
        
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
            swapChain = logicalDevice->createSwapchainKHRUnique(scCreateInfo);
        } catch (vk::SystemError err) {
            throw std::runtime_error("Failed to create swap chain." + std::string(err.what()));
        }
        
        swapChainImages = logicalDevice->getSwapchainImagesKHR(swapChain.get());

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
        vk::ImageMemoryBarrier barrier;
        barrier.setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
        barrier.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
        barrier.setImage(image);
        barrier.setOldLayout(oldLayout);
        barrier.setNewLayout(newLayout);
        barrier.setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
        barrier.setSrcAccessMask(srcAccess);
        barrier.setDstAccessMask(dstAccess);
        commandBuffer.pipelineBarrier(srcStage, dstStage, {}, {}, {}, barrier);
    }

    void createRTOutputImage()
    {
        const auto format = vk::Format::eR32G32B32A32Sfloat;
        auto imageInfo = vk::ImageCreateInfo()
            .setImageType(vk::ImageType::e2D)
            .setExtent({RT_WIDTH, RT_HEIGHT, 1})
            .setMipLevels(1)
            .setArrayLayers(1)
            .setFormat(format)
            .setUsage(vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc);
        auto tmpImage = logicalDevice->createImageUnique(imageInfo);

        vk::MemoryRequirements requirements = logicalDevice->getImageMemoryRequirements(*tmpImage);
        uint32_t memoryTypeIndex = findMemoryType(requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
        auto memoryInfo = vk::MemoryAllocateInfo()
            .setAllocationSize(requirements.size)
            .setMemoryTypeIndex(memoryTypeIndex);
        auto tmpMemory = logicalDevice->allocateMemoryUnique(memoryInfo);

        logicalDevice->bindImageMemory(*tmpImage, *tmpMemory, 0);

        auto imageViewInfo = vk::ImageViewCreateInfo()
            .setImage(*tmpImage)
            .setViewType(vk::ImageViewType::e2D)
            .setFormat(format)
            .setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
        auto tmpView = logicalDevice->createImageViewUnique(imageViewInfo);

        rtDescImageInfos.push_back(vk::DescriptorImageInfo()
            .setImageView(*tmpView)
            .setImageLayout(vk::ImageLayout::eGeneral));

        // On a besoin de créer un command buffer ici, pour qu'au moment où le gpu est prêt on fasse passer cet imageView de eUndefined à eGeneral
        auto commandBufferInfo = vk::CommandBufferAllocateInfo()
            .setCommandPool(*commandPool)
            .setCommandBufferCount(1);

        vk::UniqueCommandBuffer commandBuffer = std::move(logicalDevice->allocateCommandBuffersUnique(commandBufferInfo).front());
        vk::CommandBufferBeginInfo beginInfo {
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
        };

        commandBuffer->begin(beginInfo);
        setImageLayout(*commandBuffer, *tmpImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
        commandBuffer->end();

        vk::SubmitInfo submitInfo;
        submitInfo.setCommandBuffers(*commandBuffer);
        graphicsQueue.submit(submitInfo);
        graphicsQueue.waitIdle();

        rtImages.push_back(std::move(tmpImage));
        rtImageMemories.push_back(std::move(tmpMemory));
        rtImageViews.push_back(std::move(tmpView));
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
                swapChainImageViews.push_back(logicalDevice->createImageViewUnique(ivCreateInfo));
            }
            catch (vk::SystemError err) {
                throw std::runtime_error("Failed to create image views from swap chain image.");
            }

            // Maintenant on crée les images qui serviront d'output au raygen shader (1 par frame in flight)
            createRTOutputImage();
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
        window->getSize(window_width, window_height);
        
        while(window_width == 0 or window_height == 0)
        {
            window->getSize(window_width, window_height);
            window->waitForEvent();
        }
        
        logicalDevice->waitIdle();
        cleanupSwapChain();
        
        createSwapChain();
        createImageViews();
    }

    // Objet vulkan contenant le bytecode d'un shader en SPIR-V
    vk::UniqueShaderModule createShaderModule(const std::vector<char>& code)
    {
        vk::ShaderModuleCreateInfo smCreateInfo{
            .flags = vk::ShaderModuleCreateFlags(),
            .codeSize = code.size(),
            .pCode = reinterpret_cast<const uint32_t*>(code.data()),
        };
        
        try {
            return logicalDevice->createShaderModuleUnique(smCreateInfo);
        } catch (vk::SystemError) {
            throw std::runtime_error("Failed to create shader module.");
        }
    }
    
    void createRaytracingPipeline()
    {
        auto rgenCode  = readFile("../shaders/out/main.rgen.spv");
        auto rmissCode = readFile("../shaders/out/main.rmiss.spv");
        auto rchitCode = readFile("../shaders/out/main.rchit.spv");
        
        auto rgenModule  = createShaderModule(rgenCode);
        auto rmissModule = createShaderModule(rmissCode);
        auto rchitModule = createShaderModule(rchitCode);

        vk::PipelineShaderStageCreateInfo stages[3];

        stages[0] = vk::PipelineShaderStageCreateInfo()
            .setStage(vk::ShaderStageFlagBits::eRaygenKHR)
            .setModule(*rgenModule)
            .setPName("main");

        stages[1] = vk::PipelineShaderStageCreateInfo()
            .setStage(vk::ShaderStageFlagBits::eMissKHR)
            .setModule(*rchitModule)
            .setPName("main");

        stages[2] = vk::PipelineShaderStageCreateInfo()
            .setStage(vk::ShaderStageFlagBits::eClosestHitKHR)
            .setModule(*rmissModule)
            .setPName("main");


        shaderGroups.push_back(vk::RayTracingShaderGroupCreateInfoKHR()
            .setType(vk::RayTracingShaderGroupTypeKHR::eGeneral)
            .setGeneralShader(0)
            .setClosestHitShader(vk::ShaderUnusedKHR)
            .setAnyHitShader(vk::ShaderUnusedKHR)
            .setIntersectionShader(vk::ShaderUnusedKHR));

        shaderGroups.push_back(vk::RayTracingShaderGroupCreateInfoKHR()
            .setType(vk::RayTracingShaderGroupTypeKHR::eGeneral)
            .setGeneralShader(1)
            .setClosestHitShader(vk::ShaderUnusedKHR)
            .setAnyHitShader(vk::ShaderUnusedKHR)
            .setIntersectionShader(vk::ShaderUnusedKHR));

        shaderGroups.push_back(vk::RayTracingShaderGroupCreateInfoKHR()
            .setType(vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup)
            .setGeneralShader(vk::ShaderUnusedKHR)
            .setClosestHitShader(2)
            .setAnyHitShader(vk::ShaderUnusedKHR)
            .setIntersectionShader(vk::ShaderUnusedKHR));


        // Ici on envoie aux shaders des valeurs pour les uniform
        auto plCreateInfo = vk::PipelineLayoutCreateInfo()
            .setSetLayouts(*descriptorSetLayout);
        
        pipelineLayout = logicalDevice->createPipelineLayoutUnique(plCreateInfo);
        
        auto pipelineCreateInfo = vk::RayTracingPipelineCreateInfoKHR()
            .setStages(stages)
            .setGroups(shaderGroups)
            .setMaxPipelineRayRecursionDepth(4)
            .setLayout(*pipelineLayout);
        
        auto pipelineCreation = logicalDevice->createRayTracingPipelineKHRUnique(nullptr, nullptr, pipelineCreateInfo);
        if(pipelineCreation.result != vk::Result::eSuccess)
        {
            throw std::runtime_error("Error when creating ray tracing pipeline !");
        }

        pipeline = std::move(pipelineCreation.value);
    }
    
    void createCommandPool()
    {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);
        
        vk::CommandPoolCreateInfo cpCreateInfo {
            .flags = vk::CommandPoolCreateFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer),
            .queueFamilyIndex = queueFamilyIndices.graphicsFamily.value()
        };
        
        try {
            commandPool = logicalDevice->createCommandPoolUnique(cpCreateInfo);
        } catch (vk::SystemError err) {
            throw std::runtime_error("Failed to create command pool.");
        }
    }
    
    // Permet d'écrire les commandes qu'on souhaite dans un command buffer
    // Cette commande s'adresse au rendu sur une image de la swapChain, d'indice imageIndex
    void recordCommandBuffer(vk::CommandBuffer& commandBuffer, uint32_t imageIndex)
    {
        vk::CommandBufferBeginInfo beginInfo {};
        // cf vk::CommandBufferUsageFlagBits::eSimultaneousUse / eRenderPassContinue / eOneTimeSubmit
        
        // Commencer à ajouter des commandes à ce command buffer
        try {
            commandBuffer.begin(beginInfo);
        } catch (vk::SystemError err) {
            throw std::runtime_error("Failed to start recording commands in command buffer.");
        }
        
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, *pipeline);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR, *pipelineLayout, 0, *descriptorSets[currentFrame], nullptr);
        
        commandBuffer.traceRaysKHR(regions[0], regions[1], regions[2], {}, RT_WIDTH, RT_HEIGHT, 1u);

        vk::Image srcImage = rtImages[currentFrame].get();
        vk::Image dstImage = swapChainImages[currentFrame];

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
            vk::Filter::eNearest
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
            .commandPool = commandPool.get(),
            .level = vk::CommandBufferLevel::ePrimary, // secondary : pourrait être appelé depuis des primary
            .commandBufferCount = (uint32_t)commandBuffers.size()
        };
        
        try {
            // allocateCommandBuffers crée ici un std::vector d'un seul élément
            commandBuffers = logicalDevice->allocateCommandBuffers(allocInfo);
        } catch (vk::SystemError err) {
            throw std::runtime_error("Failed to create command buffer.");
        }
        
        for(size_t i = 0; i < commandBuffers.size(); i++)
        {
            recordCommandBuffer(commandBuffers[i], (uint32_t)i);
        }
        
    }
    
    void createSyncObjects()
    {
        // 2 sémaphores : le GPU attend qu'une image de la swap chain ait été trouvée
        // puis que le rendu ait été fini dessus pour la présenter à l'écran
        // 1 fence (= sémaphore pour le CPU) : l'exécution attend la fin de la frame
        // pour pas faire le rendu de 2 frames en même temps
        // et on fait ça pour chaque frame in flight
        
        imageAvailableSemaphores.resize(NB_FRAMES_IN_FLIGHT);
        readyForPresentationSemaphores.resize(NB_FRAMES_IN_FLIGHT);
        readyForNextFrameFences.resize(NB_FRAMES_IN_FLIGHT);
        
        vk::FenceCreateInfo fenceInfo {
            // On crée le fence en état "signalé" pour pas que le premier draw call attende indéfiniment
            .flags = vk::FenceCreateFlags(vk::FenceCreateFlagBits::eSignaled)
        };
        
        try {
            for(int i = 0; i < NB_FRAMES_IN_FLIGHT; i++)
            {
                imageAvailableSemaphores[i] = logicalDevice->createSemaphoreUnique({});
                readyForPresentationSemaphores[i] = logicalDevice->createSemaphoreUnique({});
                readyForNextFrameFences[i] = logicalDevice->createFenceUnique(fenceInfo);
            }
        } catch (vk::SystemError err) {
            throw std::runtime_error("Failed to create semaphores / fence.");
        }
    }

    void createBuffer(vk::DeviceSize size,
                      vk::BufferUsageFlags usage,
                      vk::MemoryPropertyFlags properties,
                      vk::UniqueBuffer& buffer,
                      vk::UniqueDeviceMemory& bufferMemory)
    {
        vk::BufferCreateInfo bufferInfo{
            .size = size,
            .usage = usage,
            .sharingMode = vk::SharingMode::eExclusive  // car utilisé par 1 seule family queue
        };
        
        buffer = logicalDevice->createBufferUnique(bufferInfo);
        

        vk::MemoryRequirements memReq = logicalDevice->getBufferMemoryRequirements(*buffer);
    
        vk::MemoryAllocateInfo allocInfo{
            .allocationSize = memReq.size,
            .memoryTypeIndex = findMemoryType(memReq.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)
        };
        
        bufferMemory = logicalDevice->allocateMemoryUnique(allocInfo);

        logicalDevice->bindBufferMemory(*buffer, *bufferMemory, 0);
    }
    
    void createBuffer(vk::DeviceSize size,
                      vk::BufferUsageFlags usage,
                      vk::MemoryPropertyFlags properties,
                      vk::UniqueBuffer& buffer,
                      vk::UniqueDeviceMemory& bufferMemory, 
                      vk::DeviceAddress& bufferAddress)
    {
        vk::BufferCreateInfo bufferInfo{
            .size = size,
            .usage = usage,
            .sharingMode = vk::SharingMode::eExclusive  // car utilisé par 1 seule family queue
        };
        
        buffer = logicalDevice->createBufferUnique(bufferInfo);
        

        vk::MemoryRequirements memReq = logicalDevice->getBufferMemoryRequirements(*buffer);
    
        vk::MemoryAllocateInfo allocInfo{
            .allocationSize = memReq.size,
            .memoryTypeIndex = findMemoryType(memReq.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)
        };
        
        bufferMemory = logicalDevice->allocateMemoryUnique(allocInfo);

        logicalDevice->bindBufferMemory(*buffer, *bufferMemory, 0);

        vk::BufferDeviceAddressInfoKHR bdaInfo{
            .sType = vk::StructureType::eBufferDeviceAddressInfo,
            .buffer = *buffer
        };
        bufferAddress = logicalDevice->getBufferAddressKHR(&bdaInfo);
    }
    
    void copyBuffer(vk::Buffer srcBuf, vk::Buffer dstBuf, vk::DeviceSize size)
    {
        vk::CommandBufferAllocateInfo allocInfo {
            .commandPool = commandPool.get(),
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1
        };
        
        vk::UniqueCommandBuffer commandBuffer = std::move(logicalDevice->allocateCommandBuffersUnique(allocInfo)[0]);
        
        vk::CommandBufferBeginInfo beginInfo {
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
        };
        
        commandBuffer->begin(beginInfo);
        
        vk::BufferCopy copyRegion {
            .size = size
        };
        
        commandBuffer->copyBuffer(srcBuf, dstBuf, 1, &copyRegion);
        commandBuffer->end();
        
        vk::SubmitInfo submitInfo {
            .commandBufferCount = 1,
            .pCommandBuffers = &commandBuffer.get()
        };
        
        graphicsQueue.submit(submitInfo);
        graphicsQueue.waitIdle();
    }
    
    void createVertexBuffer()
    {
        vk::DeviceSize bufSize = vertices.size() * sizeof(vertices[0]);
        
        // Un staging buffer sert à pouvoir avoir un vbuf avec VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        // mais dcp elle est pas mappable sur cpu donc il faut passer par un intermédiaire (staging buf)
        vk::UniqueBuffer stagingBuffer;
        vk::UniqueDeviceMemory stagingBufferMemory;
        
        createBuffer(bufSize, vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
                     vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostVisible,
                     stagingBuffer, stagingBufferMemory, vertexBufferDeviceAddress);

        // on met vertices dans le staging buffer
        void* data = logicalDevice->mapMemory(*stagingBufferMemory, 0, bufSize);
        memcpy(data, vertices.data(), (size_t)bufSize);
        logicalDevice->unmapMemory(*stagingBufferMemory);
        
        // on met le staging buffer dans le vrai vertex buffer
        createBuffer(bufSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, vertexBuffer, vertexBufferMemory);
        
        copyBuffer(*stagingBuffer, *vertexBuffer, bufSize);
    }
    
    void createIndexBuffer()
    {
        vk::DeviceSize bufSize = indices.size() * sizeof(indices[0]);
        
        // Un staging buffer sert à pouvoir avoir un vbuf avec VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        // mais dcp elle est pas mappable sur cpu donc il faut passer par un intermédiaire (staging buf)
        vk::UniqueBuffer stagingBuffer;
        vk::UniqueDeviceMemory stagingBufferMemory;
        
        createBuffer(bufSize, vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
                     vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostVisible,
                     stagingBuffer, stagingBufferMemory, indexBufferDeviceAddress);
        
        // on met les index dans le staging buffer
        void* data = logicalDevice->mapMemory(*stagingBufferMemory, 0, bufSize);
        memcpy(data, indices.data(), (size_t)bufSize);
        logicalDevice->unmapMemory(*stagingBufferMemory);
        
        // on met le staging buffer dans le vrai index buffer
        createBuffer(bufSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, indexBuffer, indexBufferMemory);
        
        copyBuffer(*stagingBuffer, *indexBuffer, bufSize);
    }
#if 0
    struct Accel {
        Accel() = default;
        Accel(const Context& context, vk::AccelerationStructureGeometryKHR geometry, uint32_t primitiveCount, vk::AccelerationStructureTypeKHR type) {
            vk::AccelerationStructureBuildGeometryInfoKHR buildGeometryInfo;
            buildGeometryInfo.setType(type);
            buildGeometryInfo.setFlags(vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);
            buildGeometryInfo.setGeometries(geometry);

            // Create buffer
            vk::AccelerationStructureBuildSizesInfoKHR buildSizesInfo = context.device->getAccelerationStructureBuildSizesKHR(  //
                vk::AccelerationStructureBuildTypeKHR::eDevice, buildGeometryInfo, primitiveCount);
            vk::DeviceSize size = buildSizesInfo.accelerationStructureSize;
            buffer = Buffer{context, Buffer::Type::AccelStorage, size};

            // Create accel
            vk::AccelerationStructureCreateInfoKHR accelInfo;
            accelInfo.setBuffer(*buffer.buffer);
            accelInfo.setSize(size);
            accelInfo.setType(type);
            accel = context.device->createAccelerationStructureKHRUnique(accelInfo);

            // Build
            Buffer scratchBuffer{context, Buffer::Type::Scratch, buildSizesInfo.buildScratchSize};
            buildGeometryInfo.setScratchData(scratchBuffer.deviceAddress);
            buildGeometryInfo.setDstAccelerationStructure(*accel);

            context.oneTimeSubmit([&](vk::CommandBuffer commandBuffer) {  //
                vk::AccelerationStructureBuildRangeInfoKHR buildRangeInfo;
                buildRangeInfo.setPrimitiveCount(primitiveCount);
                buildRangeInfo.setFirstVertex(0);
                buildRangeInfo.setPrimitiveOffset(0);
                buildRangeInfo.setTransformOffset(0);
                commandBuffer.buildAccelerationStructuresKHR(buildGeometryInfo, &buildRangeInfo);
            });

            descAccelInfo.setAccelerationStructures(*accel);
        }

        Buffer buffer;
        vk::UniqueAccelerationStructureKHR accel;
        vk::WriteDescriptorSetAccelerationStructureKHR descAccelInfo;
    };
#endif
    void createAccelerationStructures()
    {
        // BLAS d'abord
        auto triangleData = vk::AccelerationStructureGeometryTrianglesDataKHR()
            .setVertexFormat(vk::Format::eR32G32B32Sfloat)
            .setVertexData({vertexBufferDeviceAddress})
            .setVertexStride(sizeof(Vertex))
            .setMaxVertex(static_cast<uint32_t>(vertices.size()))
            .setIndexType(vk::IndexType::eUint32)
            .setIndexData({indexBufferDeviceAddress});

        auto triangleGeometry = vk::AccelerationStructureGeometryKHR()
            .setGeometryType(vk::GeometryTypeKHR::eTriangles)
            .setGeometry({triangleData})
            .setFlags(vk::GeometryFlagBitsKHR::eOpaque);

        const auto primitiveCount = static_cast<uint32_t>(indices.size() / 3);

        // remplir blasBuf, blasBufMemory, blasBufDeviceAddress, blasAccel, blasDescInfo
        auto buildGeometryInfo = vk::AccelerationStructureBuildGeometryInfoKHR()
            .setType(vk::AccelerationStructureTypeKHR::eBottomLevel)
            .setFlags(vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace)
            .setGeometries(triangleGeometry);

        vk::AccelerationStructureBuildSizesInfoKHR buildSizesInfo = logicalDevice->getAccelerationStructureBuildSizesKHR(
            vk::AccelerationStructureBuildTypeKHR::eDevice, buildGeometryInfo, primitiveCount);
        
        vk::DeviceSize size = buildSizesInfo.accelerationStructureSize;
        createBuffer(size, vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress, 
            vk::MemoryPropertyFlagBits::eDeviceLocal, blasBuf, blasBufMemory, blasBufDeviceAddress); 

        auto accelInfo = vk::AccelerationStructureCreateInfoKHR()
            .setBuffer(*blasBuf)
            .setSize(size)
            .setType(vk::AccelerationStructureTypeKHR::eBottomLevel);
        
        blasAccel = logicalDevice->createAccelerationStructureKHRUnique(accelInfo);

        vk::UniqueBuffer scratchBuf{};
        vk::UniqueDeviceMemory scratchBufMemory{};
        vk::DeviceAddress scratchBufDeviceAddress{};
        createBuffer(buildSizesInfo.buildScratchSize, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
            vk::MemoryPropertyFlagBits::eDeviceLocal, scratchBuf, scratchBufMemory, scratchBufDeviceAddress);

        buildGeometryInfo.setScratchData({scratchBufDeviceAddress})
                         .setDstAccelerationStructure(*blasAccel);

        // TODO : est-ce que c'est vrmt compatible avec notre command pool ?
        // on alloue un nouveau command buf qui servira à construire le blas
        // donc sera submit UNE seule fois au lancement du programme
        auto commandBufferInfo = vk::CommandBufferAllocateInfo()
            .setCommandPool(*commandPool)
            .setCommandBufferCount(1);

        vk::UniqueCommandBuffer commandBuffer = std::move(logicalDevice->allocateCommandBuffersUnique(commandBufferInfo).front());
        vk::CommandBufferBeginInfo beginInfo {
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
        };

        // on record la construction du blas
        commandBuffer->begin(beginInfo);
        auto buildRangeInfo = vk::AccelerationStructureBuildRangeInfoKHR()
            .setPrimitiveCount(primitiveCount)
            .setFirstVertex(0)
            .setPrimitiveOffset(0)
            .setTransformOffset(0);
        commandBuffer->buildAccelerationStructuresKHR(buildGeometryInfo, &buildRangeInfo);
        commandBuffer->end();

        vk::SubmitInfo submitInfo;
        submitInfo.setCommandBuffers(*commandBuffer);
        graphicsQueue.submit(submitInfo);
        graphicsQueue.waitIdle();

        blasDescInfo.setAccelerationStructures(*blasAccel);

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
            .setAccelerationStructureReference(blasBufDeviceAddress)
            .setFlags(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable);

        vk::UniqueBuffer instancesBuf{};
        vk::UniqueDeviceMemory instancesBufMemory{};
        vk::DeviceAddress instancesBufDeviceAddress{};
        createBuffer(sizeof(vk::AccelerationStructureInstanceKHR), vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
            vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostVisible, instancesBuf, instancesBufMemory, instancesBufDeviceAddress);

        auto instancesData = vk::AccelerationStructureGeometryInstancesDataKHR()
            .setArrayOfPointers(false)
            .setData({instancesBufDeviceAddress});

        auto instanceGeometry = vk::AccelerationStructureGeometryKHR()
            .setGeometryType(vk::GeometryTypeKHR::eInstances)
            .setGeometry({triangleData})
            .setFlags(vk::GeometryFlagBitsKHR::eOpaque);

        // remplir tlasBuf, tlasBufMemory, tlasBufDeviceAddress, tlasAccel, tlasDescInfo
        auto tlasBuildGeometryInfo = vk::AccelerationStructureBuildGeometryInfoKHR()
            .setType(vk::AccelerationStructureTypeKHR::eTopLevel)
            .setFlags(vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace)
            .setGeometries(instanceGeometry);

        vk::AccelerationStructureBuildSizesInfoKHR tlasBuildSizesInfo = logicalDevice->getAccelerationStructureBuildSizesKHR(
            vk::AccelerationStructureBuildTypeKHR::eDevice, tlasBuildGeometryInfo, 1);
        vk::DeviceSize tlasSize = tlasBuildSizesInfo.accelerationStructureSize;
        createBuffer(tlasSize, vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress, 
            vk::MemoryPropertyFlagBits::eDeviceLocal, tlasBuf, tlasBufMemory, tlasBufDeviceAddress); 

        auto tlasAccelInfo = vk::AccelerationStructureCreateInfoKHR()
            .setBuffer(*tlasBuf)
            .setSize(tlasSize)
            .setType(vk::AccelerationStructureTypeKHR::eTopLevel);
        tlasAccel = logicalDevice->createAccelerationStructureKHRUnique(tlasAccelInfo);

        vk::UniqueBuffer tlasScratchBuf{};
        vk::UniqueDeviceMemory tlasScratchBufMemory{};
        vk::DeviceAddress tlasScratchBufDeviceAddress{};
        createBuffer(tlasBuildSizesInfo.buildScratchSize, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
            vk::MemoryPropertyFlagBits::eDeviceLocal, tlasScratchBuf, tlasScratchBufMemory, tlasScratchBufDeviceAddress);

        tlasBuildGeometryInfo.setScratchData({tlasScratchBufDeviceAddress});
        tlasBuildGeometryInfo.setDstAccelerationStructure(*tlasAccel);

        // idem que pour la commande de construction du BLAS
        auto tlasCommandBufferInfo = vk::CommandBufferAllocateInfo()
            .setCommandPool(*commandPool)
            .setCommandBufferCount(1);

        vk::UniqueCommandBuffer tlasCommandBuffer = std::move(logicalDevice->allocateCommandBuffersUnique(tlasCommandBufferInfo).front());
        vk::CommandBufferBeginInfo tlasBeginInfo {
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
        };

        tlasCommandBuffer->begin(tlasBeginInfo);
        auto tlasBuildRangeInfo = vk::AccelerationStructureBuildRangeInfoKHR()
            .setPrimitiveCount(1) // 1 seul noeud dans notre TLAS pour l'instant (y'a 1 objet)
            .setFirstVertex(0)
            .setPrimitiveOffset(0)
            .setTransformOffset(0);
        tlasCommandBuffer->buildAccelerationStructuresKHR(tlasBuildGeometryInfo, &tlasBuildRangeInfo);
        tlasCommandBuffer->end();

        vk::SubmitInfo tlasSubmitInfo;
        tlasSubmitInfo.setCommandBuffers(*tlasCommandBuffer);
        graphicsQueue.submit(tlasSubmitInfo);
        graphicsQueue.waitIdle();

        tlasDescInfo.setAccelerationStructures(*tlasAccel);

        // pfiou, 150 lignes pour faire pousser un arbre...
    }
    
    void createUniformBuffers()
    {
        vk::DeviceSize bufSize = sizeof(UniformBufferObject);
        
        uniformBuffers.resize(NB_FRAMES_IN_FLIGHT);
        uniformBuffersMemory.resize(NB_FRAMES_IN_FLIGHT);
        uniformBuffersMapped.resize(NB_FRAMES_IN_FLIGHT);
        
        for (size_t i = 0; i < NB_FRAMES_IN_FLIGHT; i++)
        {
            createBuffer(bufSize,
                         vk::BufferUsageFlagBits::eUniformBuffer,
                         vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostVisible,
                         uniformBuffers[i],
                         uniformBuffersMemory[i]);
            
            uniformBuffersMapped[i] = logicalDevice->mapMemory(*(uniformBuffersMemory[i]), 0, bufSize);
        }
    }

    void createDescriptorSetLayout()
    {
        bindings.push_back({0, vk::DescriptorType::eAccelerationStructureKHR, 1, vk::ShaderStageFlagBits::eRaygenKHR});   // Binding = 0 : TLAS
        bindings.push_back({1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eRaygenKHR});               // Binding = 1 : Storage image
        // {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eClosestHitKHR},                        // Binding = 2 : Vertices
        // {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eClosestHitKHR}                         // Binding = 3 : Indices
        
        auto layoutInfo = vk::DescriptorSetLayoutCreateInfo()
            .setBindings(bindings);

        descriptorSetLayout = logicalDevice->createDescriptorSetLayoutUnique(layoutInfo);
    }
    
    void createDescriptorPool()
    {
        // vk::DescriptorPoolSize poolSize {
        //     .type            = vk::DescriptorType::eUniformBuffer,  // ne pas oublier cette ligne haha
        //     .descriptorCount = static_cast<uint32_t>(NB_FRAMES_IN_FLIGHT)
        // };
        
        // vk::DescriptorPoolCreateInfo poolInfo {
        //     .maxSets        = static_cast<uint32_t>(NB_FRAMES_IN_FLIGHT),
        //     .poolSizeCount  = 1,
        //     .pPoolSizes     = &poolSize
        // };

        std::vector<vk::DescriptorPoolSize> poolSizes{
            {vk::DescriptorType::eAccelerationStructureKHR, NB_FRAMES_IN_FLIGHT},
            {vk::DescriptorType::eStorageImage, NB_FRAMES_IN_FLIGHT}
            // {vk::DescriptorType::eStorageBuffer, 0 * NB_FRAMES_IN_FLIGHT}, // en vrai il en faut ((vertexbuf + indexbuf) * nb_meshes + materialsbuf) * NB_FRAMES_IN_FLIGHT
        };

        vk::DescriptorPoolCreateInfo poolInfo;
        poolInfo.setPoolSizes(poolSizes);
        poolInfo.setMaxSets(1);
        poolInfo.setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);
        
        descriptorPool = logicalDevice->createDescriptorPoolUnique(poolInfo);
    }
    
    void createDescriptorSets()
    {
#if 0
        std::vector<vk::DescriptorSetLayout> layouts(NB_FRAMES_IN_FLIGHT, *descriptorSetLayout);
        
        vk::DescriptorSetAllocateInfo allocInfo {
            .descriptorPool     = *descriptorPool,
            .descriptorSetCount = static_cast<uint32_t>(NB_FRAMES_IN_FLIGHT),
            .pSetLayouts        = layouts.data()
        };
        
        descriptorSets = logicalDevice->allocateDescriptorSets(allocInfo);
        
        for (size_t i = 0; i < NB_FRAMES_IN_FLIGHT; i++)
        {
            // Ici on met le contenu du UBO dans chaque descriptor set (un par frame-in-flight)
            vk::DescriptorBufferInfo bufInfo {
                .buffer = *uniformBuffers[i],
                .offset = 0,
                .range  = sizeof(UniformBufferObject)
            };
            
            vk::WriteDescriptorSet descriptorWrite {
                .dstSet             = descriptorSets[i],
                .dstBinding         = 0,    // quand on écrit layout(binding = 0) en GLSL on y accède ici à ce 0
                .dstArrayElement    = 0,
                .descriptorCount    = 1,
                .descriptorType     = vk::DescriptorType::eUniformBuffer,
                .pBufferInfo        = &bufInfo
            };
            
            logicalDevice->updateDescriptorSets(1, &descriptorWrite, 0, nullptr);
        }

        // __________________________________________________________________

        // Get ray tracing properties
        auto properties = context.physicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
        auto rtProperties = properties.get<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();

        // Calculate shader binding table (SBT) size
        uint32_t handleSize = rtProperties.shaderGroupHandleSize;
        uint32_t handleSizeAligned = rtProperties.shaderGroupHandleAlignment;
        uint32_t groupCount = static_cast<uint32_t>(shaderGroups.size());
        uint32_t sbtSize = groupCount * handleSizeAligned;

        // Get shader group handles
        std::vector<uint8_t> handleStorage(sbtSize);
        if (context.device->getRayTracingShaderGroupHandlesKHR(*pipeline, 0, groupCount, sbtSize, handleStorage.data()) != vk::Result::eSuccess) {
            throw std::runtime_error("failed to get ray tracing shader group handles.");
        }

        // Create SBT
        Buffer raygenSBT{context, Buffer::Type::ShaderBindingTable, handleSize, handleStorage.data() + 0 * handleSizeAligned};
        Buffer missSBT{context, Buffer::Type::ShaderBindingTable, handleSize, handleStorage.data() + 1 * handleSizeAligned};
        Buffer hitSBT{context, Buffer::Type::ShaderBindingTable, handleSize, handleStorage.data() + 2 * handleSizeAligned};

        uint32_t stride = rtProperties.shaderGroupHandleAlignment;
        uint32_t size = rtProperties.shaderGroupHandleAlignment;

        vk::StridedDeviceAddressRegionKHR raygenRegion{raygenSBT.deviceAddress, stride, size};
        vk::StridedDeviceAddressRegionKHR missRegion{missSBT.deviceAddress, stride, size};
        vk::StridedDeviceAddressRegionKHR hitRegion{hitSBT.deviceAddress, stride, size};

        // Create desc set
        vk::UniqueDescriptorSet descSet = context.allocateDescSet(*descSetLayout);
        std::vector<vk::WriteDescriptorSet> writes(bindings.size());
        for (int i = 0; i < bindings.size(); i++) {
            writes[i].setDstSet(*descSet);
            writes[i].setDescriptorType(bindings[i].descriptorType);
            writes[i].setDescriptorCount(bindings[i].descriptorCount);
            writes[i].setDstBinding(bindings[i].binding);
        }
        writes[0].setPNext(&topAccel.descAccelInfo);
        writes[1].setImageInfo(outputImage.descImageInfo);
        writes[2].setBufferInfo(vertexBuffer.descBufferInfo);
        writes[3].setBufferInfo(indexBuffer.descBufferInfo);
        writes[4].setBufferInfo(faceBuffer.descBufferInfo);
        context.device->updateDescriptorSets(writes, nullptr);
#endif
        // cher gpu, avec quel alignement mémoire dois-je construire ma shader binding table ?
        auto properties = physicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
        auto rtProperties = properties.get<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();

        uint32_t handleSize = rtProperties.shaderGroupHandleSize;
        uint32_t handleSizeAligned = rtProperties.shaderGroupHandleAlignment;
        uint32_t groupCount = static_cast<uint32_t>(shaderGroups.size()); // nb de groupes (1 pour rgen, 1 pour rmiss, 1 pour rchit pour l'instant)
        uint32_t sbtSize = groupCount * handleSizeAligned;

        // TODO : se renseigner sur shader record (comment utiliser des matériaux (ou autres params) différents par instance ?)

        // globalement ici on demande l'agencement mémoire des shaders qu'on a chargés, pour les donner aux shaders (via les descriptor sets)
        // mais pk des shaders auraient besoin de connaitre l'adresse mémoire de leur code source ? Astuce de Quine ??
        // en fait c'est parce qu'ils vont pouvoir s'appeler les uns les autres récursviement (ex : chit appelle raygen pour ombres / réflections)
        std::vector<uint8_t> handleStorage(sbtSize);
        if (logicalDevice->getRayTracingShaderGroupHandlesKHR(*pipeline, 0, groupCount, sbtSize, handleStorage.data()) != vk::Result::eSuccess) {
            throw std::runtime_error("Error when getting raytracing shader group handles.");
        }

        vk::UniqueBuffer raygenSBT{};
        vk::UniqueDeviceMemory raygenSBTMemory{};
        vk::DeviceAddress raygenSBTDeviceAddress{};
        createBuffer(handleSize, vk::BufferUsageFlagBits::eShaderBindingTableKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress,
            vk::MemoryPropertyFlagBits::eDeviceLocal, raygenSBT, raygenSBTMemory, raygenSBTDeviceAddress);
        
        // on écrit une adresse mémoire à une adresse mémoire (astuce de Quine ?)
        // par contre ce qui m'étonne c'est qu'on écrit l'adresse mémoire de son propre programme au lieu d'écrire celle des autres shaders
        void* mapped = logicalDevice->mapMemory(*raygenSBTMemory, 0, handleSize);
        memcpy(mapped, handleStorage.data() + 0 * handleSizeAligned, handleSize);
        logicalDevice->unmapMemory(*raygenSBTMemory);

        vk::UniqueBuffer rmissSBT{};
        vk::UniqueDeviceMemory rmissSBTMemory{};
        vk::DeviceAddress rmissSBTDeviceAddress{};
        createBuffer(handleSize, vk::BufferUsageFlagBits::eShaderBindingTableKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress,
            vk::MemoryPropertyFlagBits::eDeviceLocal, rmissSBT, rmissSBTMemory, rmissSBTDeviceAddress);

        void* mapped2 = logicalDevice->mapMemory(*rmissSBTMemory, 0, handleSize);
        memcpy(mapped2, handleStorage.data() + 1 * handleSizeAligned, handleSize);
        logicalDevice->unmapMemory(*rmissSBTMemory);

        vk::UniqueBuffer rchitSBT{};
        vk::UniqueDeviceMemory rchitSBTMemory{};
        vk::DeviceAddress rchitSBTDeviceAddress{};
        createBuffer(handleSize, vk::BufferUsageFlagBits::eShaderBindingTableKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress,
            vk::MemoryPropertyFlagBits::eDeviceLocal, rchitSBT, rchitSBTMemory, rchitSBTDeviceAddress);

        void* mapped3 = logicalDevice->mapMemory(*rchitSBTMemory, 0, handleSize);
        memcpy(mapped3, handleStorage.data() + 2 * handleSizeAligned, handleSize);
        logicalDevice->unmapMemory(*rchitSBTMemory);

        uint32_t stride = rtProperties.shaderGroupHandleAlignment;
        uint32_t size = rtProperties.shaderGroupHandleAlignment;

        regions.push_back({raygenSBTDeviceAddress, stride, size});
        regions.push_back({rmissSBTDeviceAddress, stride, size});
        regions.push_back({rchitSBTDeviceAddress, stride, size});

        std::vector<vk::DescriptorSetLayout> layouts(NB_FRAMES_IN_FLIGHT, *descriptorSetLayout);
        auto descSetInfo = vk::DescriptorSetAllocateInfo()
            .setDescriptorPool(*descriptorPool)
            .setSetLayouts(layouts);
        
        descriptorSets = logicalDevice->allocateDescriptorSetsUnique(descSetInfo);
        
        for(int frame = 0; frame < NB_FRAMES_IN_FLIGHT; frame++)
        {
            std::vector<vk::WriteDescriptorSet> writes(bindings.size());
            for (int i = 0; i < bindings.size(); i++) {
                writes[i].setDstSet(descriptorSets[frame].get());
                writes[i].setDescriptorType(bindings[i].descriptorType);
                writes[i].setDescriptorCount(bindings[i].descriptorCount);
                writes[i].setDstBinding(bindings[i].binding);
            }
            writes[0].setPNext(tlasDescInfo);
            writes[1].setImageInfo(rtDescImageInfos[frame]);
            logicalDevice->updateDescriptorSets(writes, nullptr);
        }
    }

    void initVulkan()
    {
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
        
        // Créer le VBO qui contient les vertex avec leurs attributs et tout
        createVertexBuffer();
        
        // Index buffer pour éviter les doublons de vertex
        createIndexBuffer();

        // Arbre qui contiendra tous nos objets (TLAS), + pour chaque objet un arbre qui stocke ses primitives (BLAS)
        createAccelerationStructures();
        
        // Contient les valeurs des uniform pour chaque frame in flight
        createUniformBuffers();
        
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


std::shared_ptr<sk::Window> sk::initWindow(unsigned int width, unsigned int height)
{
    currentFrame = 0;
    windowResized = false;
    window_width = width, window_height = height;
    window = std::make_shared<sk::Window>(width, height);

    initVulkan();

    return window;
}

void sk::setUniforms(float uTime, math::vec3 uClr)
{
    UniformBufferObject ubo = {
        .uTime  = uTime,
        .uClr   = uClr
    };
        
    memcpy(uniformBuffersMapped[currentFrame], &ubo, sizeof(ubo));
}

void sk::draw()
{
    // On commence par attendre que la frame précédente soit finie
    if(logicalDevice->waitForFences(1, &readyForNextFrameFences[currentFrame].get(), vk::True, UINT64_MAX) != vk::Result::eSuccess)
    {
        throw std::runtime_error("Timeout or error during waitForFences !");
    }
    
    // Obtient la prochaine image dispo de la swap chain, et puis fait un post dans imageAvailable
    uint32_t imgId;
    try {
        imgId = logicalDevice->acquireNextImageKHR(swapChain.get(), UINT64_MAX, imageAvailableSemaphores[currentFrame].get(), nullptr).value;
    } catch (vk::OutOfDateKHRError(const std::string &msg)) {
        recreateSwapChain();
        return;
    }
    
    // On reset le fence uniquement si on doit pas recréer la swap chain (évite une famine)
    logicalDevice->resetFences(readyForNextFrameFences[currentFrame].get());
    
    // Ensuite il faut record ce qu'on veut faire dans commandBuffer, pour l'image d'indice imgId
    commandBuffers[currentFrame].reset();
    recordCommandBuffer(commandBuffers[currentFrame], (uint32_t)imgId);
    
    // On voudra attendre le sémaphore imageAvailable au moment de la copie de rtImage sur swapChainImage
    vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eTransfer };
    
    // Ensuite on peut submit le command buffer
    vk::SubmitInfo submitInfo {
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &imageAvailableSemaphores[currentFrame].get(),
        .pWaitDstStageMask = waitStages,
        .commandBufferCount = 1,
        .pCommandBuffers = &commandBuffers[currentFrame],
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &readyForPresentationSemaphores[currentFrame].get()
    };

    graphicsQueue.submit(submitInfo, readyForNextFrameFences[currentFrame].get());
    
    // Reste plus qu'à envoyer le résultat du rendu à la swap chain pour qu'on puisse le voir
    vk::PresentInfoKHR presentInfo {
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &readyForPresentationSemaphores[currentFrame].get(),
        .swapchainCount = 1,
        .pSwapchains = &swapChain.get(),
        .pImageIndices = &imgId,
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
