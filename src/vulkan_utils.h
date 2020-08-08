//PFN_
// Created by euclid on 5/8/20.
//

#ifndef GRAPHICS_DEMO_VULKAN_UTILS_H
#define GRAPHICS_DEMO_VULKAN_UTILS_H

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <iostream>
#include <optional>
#include <cmath>
#include <unistd.h>
#include <chrono>
#include <vulkan/vulkan.hpp>
#include <random>

#include "external/lodepng.h"

using namespace vk;
using namespace std::chrono;

#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif

#define VK_CHECK_RESULT(f)                                                                                \
{                                                                                                        \
    VkResult res = (f);                                                                                    \
    if (res != VK_SUCCESS)                                                                                \
    {                                                                                                    \
        char buffer [150];                                                                              \
        sprintf(buffer, "Fatal : VkResult is %d in %s at line %d\n", res,  __FILE__, __LINE__);          \
        throw std::runtime_error(buffer);                                                                        \
    }                                                                                                    \
}

#define CHECK_RESULT(f)                                                                                \
{                                                                                                        \
    Result res = (f);                                                                                    \
    if (res != Result::eSuccess)                                                                                \
    {                                                                                                    \
        char buffer [150];                                                                              \
        sprintf(buffer, "Fatal : VkResult is not success in %s at line %d\n",  __FILE__, __LINE__);          \
        throw std::runtime_error(buffer);                                                                        \
    }                                                                                                    \
}

const std::vector<const char*> validationLayers = {
        "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;
    std::optional<uint32_t> computeFamily;

    bool isComplete() {
        return graphicsFamily.has_value() &&
               presentFamily.has_value() &&
               computeFamily.has_value();
    }

    bool isSingleFamily() {
        bool isGraphAndPres = graphicsFamily.value() == presentFamily.value();
        bool isGraphAndComp = graphicsFamily.value() == computeFamily.value() ;
        return isGraphAndPres && isGraphAndComp;
    }

    uint32_t getSingleFamilyIndex() {
        return graphicsFamily.value();
    }
};

struct SwapChainSupportDetails {
    SurfaceCapabilitiesKHR  capabilities;
    std::vector<SurfaceFormatKHR> formats;
    std::vector<PresentModeKHR> presentModes;

    bool isComplete() {
        return !formats.empty() && !presentModes.empty();
    }
};

struct BufferDetails {
    Buffer buffer;
    DeviceMemory bufferMemory;
    uint32_t bufferSize;
    int binding;
    BufferUsageFlagBits usageType;
};

struct Pixel {
    float r, g, b, a;
};

struct UniformBufferObject {
    float width;
    float height;
    float params;
    float alignment;
};

struct VkState {
    int width;
    int height;
    int workgroupSize;

    milliseconds startLoopMs;
    float params = -1.0f;

    GLFWwindow* window;
    SurfaceKHR surface;

    QueueFamilyIndices familyIndices;
    SwapChainSupportDetails swapChainSupportDetails;

    Instance instance = nullptr;
    PhysicalDevice physicalDevice = nullptr;
    Device logicalDevice = nullptr;
    Queue queue;
    SwapchainKHR swapChain;

    Format swapChainImageFormat;
    SurfaceFormatKHR surfaceFormat;
    PresentModeKHR presentMode;
    Extent2D extent2D;

    std::vector<Image> swapChainImages;
    std::vector<ImageView> swapChainImageViews;
    std::vector<Framebuffer> framebuffers;

    CommandPool commandPool;
    DescriptorPool descriptorPool;
    std::vector<CommandBuffer> commandBuffers;

    RenderPass renderPass;
    PipelineLayout graphicsPipelineLayout;
    Pipeline graphicsPipeline;
    DescriptorSetLayout graphicsDescriptorSetLayout;
    DescriptorSet graphicsDescriptorSet;

    BufferDetails dataBuffer;
    BufferDetails dataBuffer2;
    BufferDetails uniformBuffer;
    std::vector<BufferDetails*> details = {
            &dataBuffer,
            &dataBuffer2,
            &uniformBuffer
    };

    Pipeline computerPipeline;
    PipelineLayout computePipelineLayout;
    DescriptorSetLayout computeDescriptorSetLayout;
    DescriptorSet computeDescriptorSet;


    Semaphore imageAvailableSemaphore;
    Semaphore renderFinishedSemaphore;

    DebugUtilsMessengerEXT debugMessenger;

    const char *shaderComp;
    const char *shaderVert;
    const char *shaderFrag;
};

std::vector<const char *> getRequiredExtensions();

VKAPI_ATTR VkBool32  VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagBitsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
        void *pUserData);

void destroyDebugMessenger(VkState &state);

void filDebugUtilsInfo(VkDebugUtilsMessengerCreateInfoEXT &createInfoExt);

bool isDeviceSuitable(PhysicalDevice &physicalDevice, QueueFamilyIndices &indices, SurfaceKHR &surface);

void findQueueFamilies(PhysicalDevice &physicalDevice, QueueFamilyIndices &indices, SurfaceKHR &surface);

void querySwapChainSupport(VkState &state, SwapChainSupportDetails &details);

void chooseSwapSurfaceFormat(VkState &state);

void chooseSwapPresentMode(VkState &state);

void chooseSwapExtent(VkState &state);

DebugUtilsMessengerCreateInfoEXT getDebugCreateInfo();

std::vector<char> readFile(const std::string& filename);

ShaderModule createShaderModule(VkState &state, const std::vector<char> &code);

uint32_t findMemoryType(
        PhysicalDevice &physicalDevice,
        uint32_t memoryTypeBits,
        Flags<MemoryPropertyFlagBits> properties
);

DescriptorType descriptionTypeByUsageType(BufferUsageFlagBits &bits);

milliseconds now();

void initGLWindow(VkState &state) {
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    state.window = glfwCreateWindow(state.width, state.height, "Vulkan", nullptr, nullptr);
}

int isWindowShouldBeClose(VkState &state) {
    return glfwWindowShouldClose(state.window);
}

void pollEvents() {
    glfwPollEvents();
}

void clean(VkState& state) {
    for (auto detail : state.details) {
        state.logicalDevice.freeMemory(detail->bufferMemory);
        state.logicalDevice.destroyBuffer(detail->buffer);
    }

    state.logicalDevice.destroySemaphore(state.imageAvailableSemaphore);
    state.logicalDevice.destroySemaphore(state.renderFinishedSemaphore);

    state.logicalDevice.destroyCommandPool(state.commandPool);
    for (auto framebuffer : state.framebuffers) {
        state.logicalDevice.destroyFramebuffer(framebuffer);
    }

    state.logicalDevice.destroyDescriptorSetLayout(state.computeDescriptorSetLayout);
    state.logicalDevice.destroyDescriptorSetLayout(state.graphicsDescriptorSetLayout);

    state.logicalDevice.destroyDescriptorPool(state.descriptorPool);

    state.logicalDevice.destroyPipeline(state.computerPipeline);
    state.logicalDevice.destroyPipelineLayout(state.computePipelineLayout);

    state.logicalDevice.destroy(state.graphicsPipeline);
    state.logicalDevice.destroy(state.graphicsPipelineLayout);
    state.logicalDevice.destroy(state.renderPass);

    for (auto imageView: state.swapChainImageViews) {
        state.logicalDevice.destroy(imageView);
    }

    state.logicalDevice.destroy(state.swapChain);
    state.instance.destroySurfaceKHR(state.surface);
    state.logicalDevice.destroy();

    destroyDebugMessenger(state);
    state.instance.destroy();

    glfwDestroyWindow(state.window);
    glfwTerminate();
}

void createVkInstance(VkState &state) {
    ApplicationInfo appInfo(
        "Hello Triangle",
        VK_MAKE_VERSION(1, 0, 0),
        "No Engine",
        VK_MAKE_VERSION(1, 0, 0),
        VK_API_VERSION_1_2
    );

    std::vector<const char*> extensions = getRequiredExtensions();

    InstanceCreateInfo createInfo(
            {},
            &appInfo,
            (enableValidationLayers) ? validationLayers.size() : 0,
            (enableValidationLayers) ? validationLayers.data() : nullptr,
            extensions.size(),
            extensions.data()
    );

    if (enableValidationLayers) {
        auto debugCreateInfo = getDebugCreateInfo();
        createInfo.setPNext(&debugCreateInfo);
    } else {
        createInfo.setPNext(nullptr);
    }

    CHECK_RESULT(createInstance(&createInfo, nullptr, &state.instance));

    VkSurfaceKHR vkSurface;
    std::cout << "Before create? " << state.surface << "; vkSurface: " << vkSurface << std::endl;
    if (glfwCreateWindowSurface(
            (VkInstance) state.instance,
            state.window,
            nullptr,
            &vkSurface
    ) != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    } else {
        state.surface = SurfaceKHR(vkSurface);
        std::cout << "After surface? " << state.surface << "; vkSurface: " << vkSurface << std::endl;
    }

    if (enableValidationLayers) {
        auto debugCreateInfo = (VkDebugUtilsMessengerCreateInfoEXT) getDebugCreateInfo();

        auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(
                state.instance, "vkCreateDebugUtilsMessengerEXT");

        //DebugUtilsMessengerEXT
        if (func != nullptr) {
            func(state.instance, &debugCreateInfo, nullptr, (VkDebugUtilsMessengerEXT *) &state.debugMessenger);
        } else {
            CHECK_RESULT(Result::eErrorExtensionNotPresent);
        }
    }
}

DebugUtilsMessengerCreateInfoEXT getDebugCreateInfo() {
    return DebugUtilsMessengerCreateInfoEXT(
                    {},
                    DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
                    DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                    DebugUtilsMessageSeverityFlagBitsEXT::eError,
                    DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                    DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                    DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
                    reinterpret_cast<PFN_vkDebugUtilsMessengerCallbackEXT>(debugCallback)
            );
}

void pickPhysicalDevice(VkState &state) {
    uint32_t  deviceCount = 0;

    state.instance.enumeratePhysicalDevices(&deviceCount, (PhysicalDevice*) nullptr);
    std::vector<PhysicalDevice> devices(deviceCount);
    state.instance.enumeratePhysicalDevices(&deviceCount, devices.data());

    QueueFamilyIndices indices = {};
    std::cout << "device count: " << devices.size() << std::endl;
    for (auto device : devices) {
        PhysicalDeviceProperties  properties;
        device.getProperties(&properties);

        std::cout << "device name: " << properties.deviceName << std::endl;
        if (isDeviceSuitable(device, indices, state.surface)) {
            state.physicalDevice = device;
            state.familyIndices = indices;

            SwapChainSupportDetails details = {};
            querySwapChainSupport(state, details);

            if (details.isComplete()) {
                state.swapChainSupportDetails = details;
                break;
            }
        }
    }
    std::cout << "graphicsQueue: " << state.familyIndices.graphicsFamily.value() << "; pres: " << state.familyIndices.presentFamily.value() << std::endl;
}

void createLogicalDevice(VkState &state) {
    if (!state.familyIndices.isComplete() || !state.familyIndices.isSingleFamily()) {
        throw std::runtime_error("Need familyIndices before create logical device");
    }

    DeviceQueueCreateInfo queueCreateInfo(
            {},
            state.familyIndices.getSingleFamilyIndex(),
            1
    );

    float queuePriority = 1.0f;
    queueCreateInfo.setPQueuePriorities(&queuePriority);

    PhysicalDeviceFeatures deviceFeatures = {};
    DeviceCreateInfo deviceCreateInfo(
            {},
            1,
            &queueCreateInfo,
            (enableValidationLayers) ? validationLayers.size() : 0,
            (enableValidationLayers) ? validationLayers.data() : nullptr,
            deviceExtensions.size(),
            deviceExtensions.data(),
            &deviceFeatures
    );

    CHECK_RESULT(state.physicalDevice.createDevice(
            &deviceCreateInfo,
            nullptr,
            &state.logicalDevice
    ));

    state.logicalDevice.getQueue(
        state.familyIndices.getSingleFamilyIndex(),
        0,
        &state.queue
    );
}

void createBufferByDetail(
        PhysicalDevice &physicalDevice,
        Device &logicalDevice,
        BufferDetails &details
) {
    BufferCreateInfo bufferInfo(
            {},
            details.bufferSize,
            details.usageType,
            SharingMode::eExclusive
    );
    CHECK_RESULT(logicalDevice.createBuffer(
            &bufferInfo,
            nullptr,
            &details.buffer
    ));

    MemoryRequirements memoryRequirements;
    logicalDevice.getBufferMemoryRequirements(
            details.buffer,
            &memoryRequirements
    );

    uint32_t memoryType = findMemoryType(
            physicalDevice,
            memoryRequirements.memoryTypeBits,
            MemoryPropertyFlagBits::eHostCoherent | MemoryPropertyFlagBits::eHostVisible
    );
    MemoryAllocateInfo allocateInfo(memoryRequirements.size, memoryType);

    CHECK_RESULT(logicalDevice.allocateMemory(
            &allocateInfo,
            nullptr,
            &details.bufferMemory
    ))

    logicalDevice.bindBufferMemory(
            details.buffer,
            details.bufferMemory,
            0
    );
}

void createDataBuffers(VkState &state) {
    state.dataBuffer.bufferSize = sizeof(Pixel) * state.width * state.height;
    state.dataBuffer.binding = 0;
    state.dataBuffer.usageType = BufferUsageFlagBits::eStorageBuffer;

    state.dataBuffer2.bufferSize = sizeof(Pixel) * state.width * state.height;
    state.dataBuffer2.binding = 1;
    state.dataBuffer2.usageType = BufferUsageFlagBits::eStorageBuffer;

    state.uniformBuffer.bufferSize = sizeof(UniformBufferObject);
    state.uniformBuffer.binding = 2;
    state.uniformBuffer.usageType = BufferUsageFlagBits::eUniformBuffer;

    for(auto detail : state.details) {
        createBufferByDetail(
                state.physicalDevice,
                state.logicalDevice,
                *detail
        );
    }
}

void createDescriptorPool(VkState &state) {
    std::vector<DescriptorPoolSize> descriptorSizes(state.details.size());

    for (int i = 0; i < state.details.size(); i++) {
        BufferUsageFlagBits bufferType = state.details[i]->usageType;
        DescriptorType descriptorType = descriptionTypeByUsageType(bufferType);

        descriptorSizes[i] = DescriptorPoolSize(
                descriptorType,
                2
        );
    }

    DescriptorPoolCreateInfo poolCreateInfo(
            {},
            2,//???
            descriptorSizes.size(),
            descriptorSizes.data()
    );

    CHECK_RESULT(state.logicalDevice.createDescriptorPool(
            &poolCreateInfo,
            nullptr,
            &state.descriptorPool
    ));
}

DescriptorType descriptionTypeByUsageType(BufferUsageFlagBits &bufferType) {
    if (bufferType == BufferUsageFlagBits::eStorageBuffer) {
        return DescriptorType::eStorageBuffer;
    } else if (bufferType == BufferUsageFlagBits::eUniformBuffer) {
        return DescriptorType::eUniformBuffer;
    } else {
        throw std::runtime_error("Not support buffer type!");
    }
}

void createComputeDescriptorSetLayout(VkState &state) {
    std::vector<DescriptorSetLayoutBinding> descriptorSets(state.details.size());
    for (int i = 0; i < state.details.size(); i++) {
        BufferDetails detail = *state.details[i];
        DescriptorType descriptorType = descriptionTypeByUsageType(detail.usageType);

        DescriptorSetLayoutBinding descriptorBinding(
                detail.binding,
                descriptorType,
                1,
                ShaderStageFlagBits::eCompute
        );

        descriptorSets[i] = descriptorBinding;
    }

    DescriptorSetLayoutCreateInfo layoutInfo(
            {},
            descriptorSets.size(),
            descriptorSets.data()
    );

    CHECK_RESULT(state.logicalDevice.createDescriptorSetLayout(
            &layoutInfo,
            nullptr,
            &state.computeDescriptorSetLayout
    ));
}

void createGraphicsDescriptorSetLayout(VkState &state) {
    std::vector<DescriptorSetLayoutBinding> descriptorSets(state.details.size());
    for (int i = 0; i < state.details.size(); i++) {
        BufferDetails detail = *state.details[i];
        DescriptorType descriptorType = descriptionTypeByUsageType(detail.usageType);
        descriptorSets[i] = DescriptorSetLayoutBinding(
                detail.binding,
                descriptorType,
                1,
                ShaderStageFlagBits::eFragment
        );
    }

    DescriptorSetLayoutCreateInfo layoutInfo(
            {},
            descriptorSets.size(),
            descriptorSets.data()
    );

    CHECK_RESULT(state.logicalDevice.createDescriptorSetLayout(
            &layoutInfo,
            nullptr,
            &state.graphicsDescriptorSetLayout
    ));
}

void createGraphicsDescriptorSet(VkState &state) {
    DescriptorSetAllocateInfo allocateInfo(
            state.descriptorPool,
            1,
            &state.graphicsDescriptorSetLayout
    );

    CHECK_RESULT(state.logicalDevice.allocateDescriptorSets(
            &allocateInfo,
            &state.graphicsDescriptorSet
    ));

    std::vector<WriteDescriptorSet> writeDescriptorSets(state.details.size());
    std::vector<std::shared_ptr<DescriptorBufferInfo>> buffHolders(state.details.size());
    for (int i = 0; i < state.details.size(); i++) {
        auto detail = *state.details[i];
        auto descriptionType = descriptionTypeByUsageType(detail.usageType);

        auto bufferInfo = std::make_shared<DescriptorBufferInfo>(
                detail.buffer,
                0,
                detail.bufferSize
        );

        buffHolders[i] = bufferInfo;
        writeDescriptorSets[i] = WriteDescriptorSet(
                state.graphicsDescriptorSet,
                detail.binding,
                {},
                1,
                descriptionType,
                {},
                bufferInfo.get()
        );
    }

    state.logicalDevice.updateDescriptorSets(
            writeDescriptorSets.size(),
            writeDescriptorSets.data(),
            0,
            nullptr
    );
}

void createComputeDescriptorSet(VkState &state) {
    DescriptorSetAllocateInfo allocateInfo(
        state.descriptorPool,
        1,
        &state.computeDescriptorSetLayout
    );

    CHECK_RESULT(state.logicalDevice.allocateDescriptorSets(
            &allocateInfo,
            &state.computeDescriptorSet
    ));

    std::vector<WriteDescriptorSet> writeDescriptorSets(state.details.size());
    std::vector<std::shared_ptr<DescriptorBufferInfo>> bufferHolders(state.details.size());
    for (int i = 0; i < state.details.size(); i++) {
        BufferDetails detail = *state.details[i];
        DescriptorType descriptorType = descriptionTypeByUsageType(detail.usageType);
        std::cout << "detail usage: " << (int)detail.usageType << "; type: " << (int)descriptorType << "; binding: " << detail.binding << std::endl;

        bufferHolders[i] = std::make_shared<DescriptorBufferInfo>(
                detail.buffer,
                0,
                detail.bufferSize
        );

        WriteDescriptorSet writeDescriptorSet(
                state.computeDescriptorSet,
                detail.binding,
                {},
                1,
                descriptorType,
                {},
                bufferHolders[i].get()
        );
        writeDescriptorSets[i] = writeDescriptorSet;
    }

    state.logicalDevice.updateDescriptorSets(
            writeDescriptorSets.size(),
            writeDescriptorSets.data(),
            0,
            nullptr
    );
}

void createComputePipeline(VkState &state) {
    auto compShaderCode = readFile(state.shaderComp);

    ShaderModule compShaderModule = createShaderModule(state, compShaderCode);
    PipelineShaderStageCreateInfo shaderStageCreateInfo(
            {},
            ShaderStageFlagBits::eCompute,
            compShaderModule,
            "main"
    );

    PipelineLayoutCreateInfo pipelineLayoutCreateInfo(
            {},
            1,
            &state.computeDescriptorSetLayout
    );

    CHECK_RESULT(state.logicalDevice.createPipelineLayout(
            &pipelineLayoutCreateInfo,
            nullptr,
            &state.computePipelineLayout
    ));

    ComputePipelineCreateInfo pipelineCreateInfo(
            {},
            shaderStageCreateInfo,
            state.computePipelineLayout
    );

    CHECK_RESULT(state.logicalDevice.createComputePipelines(
            nullptr,
            1,
            &pipelineCreateInfo,
            nullptr,
            &state.computerPipeline
    ));

    state.logicalDevice.destroyShaderModule(compShaderModule);
}

uint32_t findMemoryType(
        PhysicalDevice &physicalDevice,
        uint32_t memoryTypeBits,
        Flags<MemoryPropertyFlagBits> properties
) {
    PhysicalDeviceMemoryProperties memoryProperties;
    physicalDevice.getMemoryProperties(&memoryProperties);

    /*
    How does this search work?
    See the documentation of VkPhysicalDeviceMemoryProperties for a detailed description.
    */
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++) {
        if ((memoryTypeBits & (1 << i)) && (memoryProperties.memoryTypes[i].propertyFlags & properties)) {
            return i;
        }
    }

    throw std::runtime_error("Can't find need memory type!");
}

void createSwapChain(VkState &state) {
    chooseSwapSurfaceFormat(state);
    chooseSwapPresentMode(state);
    chooseSwapExtent(state);

    uint32_t imageCount = state.swapChainSupportDetails.capabilities.minImageCount + 1;

    SwapchainCreateInfoKHR swapchainCreateInfo;

    swapchainCreateInfo.surface = state.surface;

    swapchainCreateInfo.minImageCount = imageCount;
    swapchainCreateInfo.imageFormat = state.surfaceFormat.format;
    swapchainCreateInfo.imageColorSpace = state.surfaceFormat.colorSpace;
    swapchainCreateInfo.imageExtent = state.extent2D;
    swapchainCreateInfo.imageArrayLayers = 1;
    swapchainCreateInfo.imageUsage = ImageUsageFlagBits::eColorAttachment;

    swapchainCreateInfo.imageSharingMode = SharingMode::eExclusive;
    swapchainCreateInfo.queueFamilyIndexCount = 0;
    swapchainCreateInfo.pQueueFamilyIndices = nullptr;

    swapchainCreateInfo.preTransform = state.swapChainSupportDetails.capabilities.currentTransform;
    swapchainCreateInfo.compositeAlpha = CompositeAlphaFlagBitsKHR::eOpaque;
    swapchainCreateInfo.presentMode = state.presentMode;
    swapchainCreateInfo.oldSwapchain = nullptr;
    swapchainCreateInfo.clipped = true;

    CHECK_RESULT(state.logicalDevice.createSwapchainKHR(
            &swapchainCreateInfo,
            nullptr,
            &state.swapChain
    ));

    state.logicalDevice.getSwapchainImagesKHR(state.swapChain, &imageCount, (Image*)(nullptr));
    state.swapChainImages.resize(imageCount);
    state.logicalDevice.getSwapchainImagesKHR(state.swapChain, &imageCount, state.swapChainImages.data());

    state.swapChainImageFormat = state.surfaceFormat.format;

    state.swapChainImageViews.resize(imageCount);
    for (int i = 0; i < imageCount; i++) {
        ImageViewCreateInfo imageViewCreateInfo = {};
        imageViewCreateInfo.image = state.swapChainImages[i];
        imageViewCreateInfo.viewType = ImageViewType::e2D;
        imageViewCreateInfo.format = state.swapChainImageFormat;
        imageViewCreateInfo.components.r = ComponentSwizzle::eIdentity;
        imageViewCreateInfo.components.g = ComponentSwizzle::eIdentity;
        imageViewCreateInfo.components.b = ComponentSwizzle::eIdentity;
        imageViewCreateInfo.components.a = ComponentSwizzle::eIdentity;
        imageViewCreateInfo.subresourceRange.aspectMask = ImageAspectFlagBits::eColor;
        imageViewCreateInfo.subresourceRange.baseMipLevel = 0;
        imageViewCreateInfo.subresourceRange.levelCount = 1;
        imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
        imageViewCreateInfo.subresourceRange.layerCount = 1;

        CHECK_RESULT(state.logicalDevice.createImageView(
            &imageViewCreateInfo,
            nullptr,
            &state.swapChainImageViews[i]
        ));
    }
}

void createRenderPass(VkState &state) {
    AttachmentDescription colorAttachment(
            {},
            state.swapChainImageFormat,
            SampleCountFlagBits::e1,
            AttachmentLoadOp::eClear,
            AttachmentStoreOp::eStore,
            AttachmentLoadOp::eDontCare,
            AttachmentStoreOp::eDontCare,
            ImageLayout::eUndefined,
            ImageLayout::ePresentSrcKHR
    );

    AttachmentReference colorAttachmentRef(
            0,
            ImageLayout::eColorAttachmentOptimal
    );

    SubpassDescription subpass(
            {},
            PipelineBindPoint::eGraphics,
            0,
            {},
            1,
            &colorAttachmentRef,
            {},
            {},
            0,
            {}
    );

    RenderPassCreateInfo renderPassCreateInfo(
            {},
            1,
            &colorAttachment,
            1,
            &subpass
    );

    SubpassDependency dependency = {};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;

    dependency.srcStageMask = PipelineStageFlagBits::eColorAttachmentOutput;
    dependency.srcAccessMask = AccessFlagBits::eColorAttachmentRead;//?????
    dependency.dstStageMask = PipelineStageFlagBits::eColorAttachmentOutput;
    dependency.dstAccessMask = AccessFlagBits::eColorAttachmentWrite;

    renderPassCreateInfo.dependencyCount = 1;
    renderPassCreateInfo.pDependencies = &dependency;

    CHECK_RESULT(state.logicalDevice.createRenderPass(&renderPassCreateInfo, nullptr, &state.renderPass));
}

void createGraphicsPipeline(VkState &state) {
    auto vertShaderCode = readFile(state.shaderVert);
    auto fragShaderCode = readFile(state.shaderFrag);

    ShaderModule vertShaderModule = createShaderModule(state, vertShaderCode);
    ShaderModule fragShaderModule = createShaderModule(state, fragShaderCode);

    PipelineShaderStageCreateInfo shaderStages[] = {
            PipelineShaderStageCreateInfo(
                    {},
                    ShaderStageFlagBits::eVertex,
                    vertShaderModule,
                    "main"
            ),
            PipelineShaderStageCreateInfo(
                    {},
                    ShaderStageFlagBits::eFragment,
                    fragShaderModule,
                    "main"
            )
    };

    PipelineVertexInputStateCreateInfo  vertexInputInfo(
            {},
            0,
            nullptr,
            0,
            nullptr
    );

    PipelineInputAssemblyStateCreateInfo inputAssembly(
            {},
            PrimitiveTopology::eTriangleList,
            false
    );

    Viewport viewport(
            0.0f,
            0.0f,
            state.extent2D.width,
            state.extent2D.height,
            0.0f,
            1.0
    );

    Rect2D scissor({0, 0}, state.extent2D);

    PipelineViewportStateCreateInfo viewportState(
            {},
            1,
            &viewport,
            1,
            &scissor
    );

    PipelineRasterizationStateCreateInfo rasterizer(
            {},
            false,
            false,
            PolygonMode::eFill,
            CullModeFlagBits::eBack,
            FrontFace::eClockwise,
            false,
            0.0f,
            0.0f,
            0.0f,
            1.0f
    );

    PipelineMultisampleStateCreateInfo multisampling(
            {},
            SampleCountFlagBits::e1,
            false,
            1.0f,
            nullptr,
            false,
            false
    );

    PipelineColorBlendAttachmentState colorBlendAttachment(
            false,
            BlendFactor::eZero,
            BlendFactor::eZero,
            BlendOp::eAdd,
            BlendFactor::eZero,
            BlendFactor::eZero,
            BlendOp::eAdd,
            ColorComponentFlagBits::eR |
            ColorComponentFlagBits::eG |
            ColorComponentFlagBits::eB |
            ColorComponentFlagBits::eA
    );

    PipelineColorBlendStateCreateInfo colorBlending(
            {},
            false,
            LogicOp::eCopy,
            1,
            &colorBlendAttachment,
            std::array<float, 4> {
                0.0f,
                0.0f,
                0.0f,
                0.0f
            }
    );

    std::array<DynamicState, 2> dynamicStates = {
            DynamicState::eViewport,
            DynamicState::eLineWidth
    };

    PipelineDynamicStateCreateInfo dynamicState(
            {},
            dynamicStates.size(),
            dynamicStates.data()
    );

    PipelineLayoutCreateInfo pipelineLayoutCreateInfo(
            {},
            1,
            &state.graphicsDescriptorSetLayout,
            0,
            nullptr
    );

    CHECK_RESULT(state.logicalDevice.createPipelineLayout(
            &pipelineLayoutCreateInfo,
            nullptr,
            &state.graphicsPipelineLayout));

    GraphicsPipelineCreateInfo pipelineInfo = {};
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = nullptr;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = nullptr;
    pipelineInfo.layout = state.graphicsPipelineLayout;
    pipelineInfo.renderPass = state.renderPass;
    pipelineInfo.subpass = 0;

    CHECK_RESULT(state.logicalDevice.createGraphicsPipelines(
            nullptr,
            1,
            &pipelineInfo,
            nullptr,
            &state.graphicsPipeline
    ));

    std::cout << shaderStages << std::endl;

    state.logicalDevice.destroyShaderModule(vertShaderModule);
    state.logicalDevice.destroyShaderModule(fragShaderModule);

    state.framebuffers.resize(state.swapChainImages.size());
    for (int i = 0; i < state.framebuffers.size(); i++) {
        std::array<ImageView, 1> attachments = { state.swapChainImageViews[i] };
        FramebufferCreateInfo framebufferInfo(
                {},
                state.renderPass,
                attachments.size(),
                attachments.data(),
                state.extent2D.width,
                state.extent2D.height,
                1
        );

        CHECK_RESULT(state.logicalDevice.createFramebuffer(
                &framebufferInfo,
                nullptr,
                &state.framebuffers[i]
        ));
    }
}

void createCommandBuffers(VkState &state) {
    CommandPoolCreateInfo poolInfo(
            {},
            state.familyIndices.getSingleFamilyIndex()
    );

    CHECK_RESULT(state.logicalDevice.createCommandPool(
        &poolInfo,
        nullptr,
        &state.commandPool
    ));

    state.commandBuffers.resize(state.framebuffers.size());
    CommandBufferAllocateInfo allocateInfo(
        state.commandPool,
        CommandBufferLevel::ePrimary,
        static_cast<uint32_t>(state.commandBuffers.size())
    );

    CHECK_RESULT(state.logicalDevice.allocateCommandBuffers(
            &allocateInfo,
            state.commandBuffers.data()
    ));

    for (int i = 0; i < state.commandBuffers.size(); i++) {
        CommandBuffer commandBuffer = state.commandBuffers[i];
        CommandBufferBeginInfo beginInfo({},nullptr);

        CHECK_RESULT(commandBuffer.begin(&beginInfo));

        commandBuffer.bindPipeline(
                PipelineBindPoint::eCompute,
                state.computerPipeline
        );
        commandBuffer.bindDescriptorSets(
                PipelineBindPoint::eCompute,
                state.computePipelineLayout,
                0,
                1,
                &state.computeDescriptorSet,
                0,
                nullptr
        );

        commandBuffer.dispatch(
                (uint32_t) std::ceil(float(state.width) / float(state.workgroupSize)),
                (uint32_t) std::ceil(float(state.height) / float(state.workgroupSize)),
                1
        );

        ClearValue clearColor = {};
        clearColor.setColor(ClearColorValue(
                std::array<float, 4>(
                        {0.0f, 0.0f, 0.0f, 1.0}
                )));

        RenderPassBeginInfo renderPassInfo(
                state.renderPass,
                state.framebuffers[i],
                Rect2D({0, 0}, state.extent2D),
                1,
                &clearColor
        );

        commandBuffer.beginRenderPass(
                &renderPassInfo,
                SubpassContents::eInline
        );
        commandBuffer.bindPipeline(
                PipelineBindPoint::eGraphics,
                state.graphicsPipeline
        );
        commandBuffer.bindDescriptorSets(
                PipelineBindPoint::eGraphics,
                state.graphicsPipelineLayout,
                0,
                1,
                &state.graphicsDescriptorSet,
                0,
                nullptr
        );
        commandBuffer.draw(
                6,
                1,
                0,
                0
        );
        commandBuffer.endRenderPass();

        commandBuffer.end();
    }
}

void createSyncObjects(VkState &state) {
    SemaphoreCreateInfo semaphoreInfo;
    CHECK_RESULT(state.logicalDevice.createSemaphore(
            &semaphoreInfo,
            nullptr,
            &state.imageAvailableSemaphore
    ));
    CHECK_RESULT(state.logicalDevice.createSemaphore(
            &semaphoreInfo,
            nullptr,
            &state.renderFinishedSemaphore
    ));
}

void drawFrame(VkState &state) {
    uint32_t  imageIndex;
    state.logicalDevice.acquireNextImageKHR(
            state.swapChain,
            UINT64_MAX,
            state.imageAvailableSemaphore,
            nullptr,
            &imageIndex
    );

    Semaphore waitSemaphores[] = {state.imageAvailableSemaphore};
    PipelineStageFlags waitStages[] = {PipelineStageFlagBits::eColorAttachmentOutput};
    Semaphore signalSemaphores[] = {state.renderFinishedSemaphore};
    SubmitInfo submitInfo(
            1,
            waitSemaphores,
            waitStages,
            1,
            &state.commandBuffers[imageIndex],
            1,
            signalSemaphores
    );

    CHECK_RESULT(
            state.queue.submit(1, &submitInfo, nullptr)
    );

    SwapchainKHR swapChains[] = { state.swapChain };
    PresentInfoKHR presentInfo(
        1,
        signalSemaphores,
        1,
        swapChains,
        &imageIndex,
        nullptr
    );
    state.queue.presentKHR(&presentInfo);
    state.queue.waitIdle();
}

void updateUniformBuffer(VkState &state) {
    UniformBufferObject ubo = {};
    ubo.width = float(state.width);
    ubo.height = float(state.height);
    ubo.alignment = 0;
    ubo.params = 0;

    milliseconds moment = now();
    double duration = double((moment - state.startLoopMs).count());
    if (duration > 25) {
        state.params = (state.params > 0.0) ? -1.0f : 1.0f;
        state.startLoopMs = moment;
        ubo.alignment = 1.0;
    } else {
        ubo.alignment = -1.0;
    }

    ubo.params = state.params;

    void *mappedMemory = nullptr;
    DeviceSize offset = 0;
    DeviceSize size = state.uniformBuffer.bufferSize;
    state.logicalDevice.mapMemory(
            state.uniformBuffer.bufferMemory,
            offset,
            size,
            {},
            &mappedMemory
    );
    memcpy(mappedMemory, &ubo, state.uniformBuffer.bufferSize);
    state.logicalDevice.unmapMemory(state.uniformBuffer.bufferMemory);
}

void initBuffers(VkState &state) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    //TODO: Не могу создать такой большой массив на стеке
    //auto *pixels = new Pixel[state.width * state.height]();
    auto pixels = std::make_unique<Pixel*>(new Pixel[state.width * state.height]);

    for (int i = 0; i < state.height; i++) {
        for (int j = 0; j < state.width; j++) {
            //auto pixel = pixels[i * state.width + j];
            auto pixel = &(*pixels)[i * state.width + j];
            pixel->r = dist(mt);
            pixel->g = 0.0;
            pixel->b = 0.0;
            pixel->a = 0.0;
        }
    }

    BufferDetails details = state.dataBuffer2;
    void *mappedMemory = nullptr;
    DeviceSize offset = 0;
    DeviceSize size = details.bufferSize;
    state.logicalDevice.mapMemory(
            details.bufferMemory,
            offset,
            size,
            {},
            &mappedMemory
    );
    memcpy(mappedMemory, (*pixels), details.bufferSize);
    state.logicalDevice.unmapMemory(details.bufferMemory);
    state.startLoopMs = now();
}

milliseconds now() {
    return duration_cast<milliseconds >(
            system_clock::now().time_since_epoch()
    );
}

void saveBuffer(VkState &state, const char* fileName) {
    void *mappedMemory = nullptr;
    DeviceSize offset = 0;
    DeviceSize size = state.dataBuffer.bufferSize;
    state.logicalDevice.mapMemory(
            state.dataBuffer.bufferMemory,
            offset,
            size,
            {},
            &mappedMemory
    );

    auto *pMappedMemory = (Pixel*) mappedMemory;
    std::vector<unsigned  char> image;
    int width = state.width;
    int height = state.height;
    int channels = 4;
    float mult = 255.0f;
    image.reserve(width * height * channels);

    for (int i = 0; i < width * height; i++) {
        float r = pMappedMemory[i].r;
        float g = pMappedMemory[i].g;
        float b = pMappedMemory[i].b;
        float a = pMappedMemory[i].a;

        image.push_back((unsigned char)(mult * r));
        image.push_back((unsigned char)(mult * g));
        image.push_back((unsigned char)(mult * b));
        image.push_back((unsigned char)(mult * a));
    }

    state.logicalDevice.unmapMemory(state.dataBuffer.bufferMemory);
    unsigned error = lodepng::encode(fileName, image, width, height);
    if (error) printf("encoder error %d: %s", error, lodepng_error_text(error));
}

void waitIdle(VkState &state) {
    state.logicalDevice.waitIdle();
}

bool isDeviceSuitable(PhysicalDevice &physicalDevice, QueueFamilyIndices &indices, SurfaceKHR &surface) {
    findQueueFamilies(physicalDevice, indices, surface);
    return indices.isComplete();
}

void findQueueFamilies(
        PhysicalDevice &physicalDevice,
        QueueFamilyIndices &indices,
        SurfaceKHR &surface) {

    uint32_t queueFamilyCount = 0;
    physicalDevice.getQueueFamilyProperties(&queueFamilyCount, (QueueFamilyProperties*) nullptr);
    std::vector<QueueFamilyProperties> queueFamilies(queueFamilyCount);
    physicalDevice.getQueueFamilyProperties(&queueFamilyCount, queueFamilies.data());

    Bool32  presentSupport = false;
    for (int i = 0; i < queueFamilies.size(); i++) {
        auto queueFamily = queueFamilies[i];
        if (queueFamily.queueFlags & QueueFlagBits::eGraphics) {
            indices.graphicsFamily = i;
        }

        if (queueFamily.queueFlags & QueueFlagBits::eCompute) {
            indices.computeFamily = i;
        }

        physicalDevice.getSurfaceSupportKHR(i, surface, &presentSupport);
        if (presentSupport) {
            indices.presentFamily = i;
        }

        if (indices.isComplete()) {
            break;
        }
    }
}

void destroyDebugMessenger(VkState &state) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(
            state.instance, "vkDestroyDebugUtilsMessengerEXT");

    if (func != nullptr) {
        func(state.instance, state.debugMessenger, nullptr);
    } else {
        throw std::runtime_error("Can't destroy debug messenger!");
    }
}

bool isPressEscape(VkState &state) {
    bool down = false;
    down = glfwGetKey(state.window, (int)GLFW_KEY_ESCAPE) == 1;
    return down;
}

std::vector<const char*> getRequiredExtensions() {
    uint32_t  glfwExtensionCount = 0;
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    if (enableValidationLayers) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    for (int i = 0; i < extensions.size(); i++) {
        std::cout << extensions[i] << std::endl;
    }

    return extensions;
}

void querySwapChainSupport(VkState &state, SwapChainSupportDetails &details) {
    state.physicalDevice.getSurfaceCapabilitiesKHR(state.surface, &details.capabilities);

    uint32_t formatCount;
    state.physicalDevice.getSurfaceFormatsKHR(state.surface,&formatCount, (SurfaceFormatKHR*) nullptr);
    details.formats.resize(formatCount);
    state.physicalDevice.getSurfaceFormatsKHR(state.surface,&formatCount, details.formats.data());

    uint32_t presentModeCount;
    state.physicalDevice.getSurfacePresentModesKHR(state.surface, &presentModeCount, (PresentModeKHR*) nullptr);
    details.presentModes.resize(presentModeCount);
    state.physicalDevice.getSurfacePresentModesKHR(state.surface, &presentModeCount, details.presentModes.data());

    if (details.formats.empty()) {
        throw std::runtime_error("Can't make swap chain without formats!");
    }

    std::cout << "surfaceFormat support count: " << details.formats.size() << std::endl;
}

void chooseSwapSurfaceFormat(VkState &state) {
    for (const auto& surfaceFormatKhr : state.swapChainSupportDetails.formats) {
        if (surfaceFormatKhr.format == Format::eB8G8R8Srgb &&
            surfaceFormatKhr.colorSpace == ColorSpaceKHR::eSrgbNonlinear) {
            state.surfaceFormat = surfaceFormatKhr;
        }
    }
    state.surfaceFormat = state.swapChainSupportDetails.formats[0];
}

void chooseSwapPresentMode(VkState &state) {
    for (auto presentMode : state.swapChainSupportDetails.presentModes) {
        if (presentMode == PresentModeKHR::eMailbox) {
            state.presentMode = presentMode;
            return;
        }
    }

    //std::cout << "Bad presentation mode :( !" << std::endl;
    state.presentMode = PresentModeKHR::eFifo;
}

void chooseSwapExtent(VkState &state) {
    if (state.swapChainSupportDetails.capabilities.currentExtent.width != UINT32_MAX) {
        state.extent2D = state.swapChainSupportDetails.capabilities.currentExtent;
        return;
    }

    throw std::runtime_error("Not supported simple extent!");
}

std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }

    size_t fileSize = (size_t) file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();

    return buffer;
}

ShaderModule createShaderModule(VkState &state, const std::vector<char> &code) {
    ShaderModuleCreateInfo createInfo(
            {},
            code.size(),
            reinterpret_cast<const uint32_t *>(code.data())
    );

    ShaderModule shaderModule;
    CHECK_RESULT(state.logicalDevice.createShaderModule(&createInfo, nullptr, &shaderModule))
    return shaderModule;
}

void delay(unsigned milliseconds) {
    usleep(milliseconds * 1000); // takes microseconds
}

VKAPI_ATTR VkBool32  VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagBitsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
        void *pUserData) {
    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

    return VK_FALSE;
}

#endif //GRAPHICS_DEMO_VULKAN_UTILS_H
