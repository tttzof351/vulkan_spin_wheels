#define GLFW_INCLUDE_VULKAN

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
#include <experimental/filesystem>
#include <vector>

#include "vulkan_utils.h"
#include "tests.h"


int main() {
    VkState state = {
            .width = 800,
            .height = 800,
            .workgroupSize = 32
    };

    state.shaderComp = "../shaders/simple_comp.spv";
    state.shaderVert = "../shaders/simple_vert.spv";
    state.shaderFrag = "../shaders/simple_frag.spv";

    initGLWindow(state);
    createVkInstance(state);
    pickPhysicalDevice(state);
    createLogicalDevice(state);

    createDataBuffers(state);
    createDescriptorPool(state);

    createComputeDescriptorSetLayout(state);
    createComputeDescriptorSet(state);
    createComputePipeline(state);

    createGraphicsDescriptorSetLayout(state);
    createGraphicsDescriptorSet(state);

    createSwapChain(state);
    createRenderPass(state);
    createGraphicsPipeline(state);

    createCommondBuffers(state);
    createSyncObjects(state);

    initBuffers(state);
    while (!isWindowShouldBeClose(state)) {
        pollEvents();
        if (isPressEscape(state)) {
            break;
        }
        updateUniformBuffer(state);
        drawFrame(state);
    }

    waitIdle(state);

    clean(state);
    return EXIT_SUCCESS;
}