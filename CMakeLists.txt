cmake_minimum_required(VERSION 3.10)
include("cmake/HunterGate.cmake")
HunterGate(
        URL "https://github.com/ruslo/hunter/archive/v0.23.111.tar.gz"
        SHA1 "4189bbd869abe4eb69b5303d3f916c03d34dc154"
)
project(vulkan_spin_wheels)

set(CMAKE_CXX_STANDARD 17)

find_package(Vulkan)

hunter_add_package(glm)
find_package(glm REQUIRED)

hunter_add_package(glew)
find_package(glew CONFIG REQUIRED)

hunter_add_package(glfw)
find_package(glfw3 REQUIRED)

add_executable(vulkan_spin_wheels src/main.cpp src/vulkan_utils.h src/external/lodepng.cpp src/external/lodepng.h src/tests.h)
target_link_libraries(vulkan_spin_wheels PRIVATE glm)
target_link_libraries(vulkan_spin_wheels PRIVATE glew::glew)
target_link_libraries(vulkan_spin_wheels PRIVATE glfw)
target_link_libraries(vulkan_spin_wheels PRIVATE Vulkan::Vulkan)
