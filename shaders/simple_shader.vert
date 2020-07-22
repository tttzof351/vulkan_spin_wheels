#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) out vec3 fragColor;

vec2 positions[6] = vec2[](
    vec2(-1.0, -1.0),//1
    vec2(1.0, -1.0),//2
    vec2(1.0, 1.0),//3
    vec2(-1.0, -1.0),//1
    vec2(1.0, 1.0),//3
    vec2(-1.0, 1.0)//4
);

vec3 colors[6] = vec3[](
    vec3(1.0, 0.0, 0.0),//1
    vec3(0.0, 1.0, 0.0),//2
    vec3(0.0, 0.0, 1.0),//3
    vec3(0.5, 0.5, 0.0),//4
    vec3(0.0, 1.0, 0.0),//2
    vec3(0.0, 0.0, 1.0)//3
);

void main() {
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    fragColor = colors[gl_VertexIndex];
}