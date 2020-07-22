#!/bin/bash

glslc ./shaders/simple_shader.vert -o ./shaders/simple_vert.spv
glslc ./shaders/simple_shader.frag -o ./shaders/simple_frag.spv
glslc ./shaders/simple_shader.comp -o ./shaders/simple_comp.spv

