#version 450
#extension GL_ARB_separate_shader_objects : enable

#define M_PI 3.1415926535897932384626433832795


struct Pixel {
    vec4 value;
};

layout(std140, binding = 0) buffer buf {
    Pixel image[];
};

layout(std140, binding = 1) buffer buf2 {
    Pixel image2[];
};

layout(binding = 2) uniform UniformBufferObject {
    float width;
    float height;
    float params;
    float alignment;
} ubo;

layout(location = 0) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

uint width = uint(ubo.width);
uint height = uint(ubo.height);

uint half_height = height/2;
uint half_width = width/2;

vec3 a = vec3(0.5, 0.5, 0.5);
vec3 b = vec3(0.5, 0.5, 0.5);
vec3 c = vec3(2.0, 1.0, 0.0);
vec3 d = vec3(0.50, 0.20, 0.25);

//https://thebookofshaders.com/06/?lan=ru
vec3 hsb2rgb( in vec3 c ){
    vec3 rgb = clamp(abs(mod(c.x*6.0+vec3(0.0,4.0,2.0), 6.0)-3.0)-1.0, 0.0, 1.0);
    rgb = rgb*rgb*(3.0-2.0*rgb);
    return c.z * mix(vec3(1.0), rgb, c.y);
}

//https://iquilezles.org/www/articles/palettes/palettes.htm
vec3 pallete(in float t) {
    return a + b * cos(M_PI * (c * t + d));
}

void main() {
    vec4 coord = gl_FragCoord;

    int x = int(coord.x);
    int y = int(coord.y);

    if (x < 0 || y < 0) {
        outColor = vec4(0.9, 0.6, 0.03, 1.0);
        return;
    } else if (x >= width || y >= height) {
        outColor = vec4(0.0, 0.0, 1.0, 1.0);
        return;
    }

    float t;
    if (ubo.params > 0.0) {
        t = image[y * width + x].value.x;
    } else {
        t = image2[y * width + x].value.x;
    }

    vec3 color = pallete(t);
    outColor = vec4(color, 1.0);
}
