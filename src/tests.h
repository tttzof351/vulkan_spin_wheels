//
// Created by euclid on 5/10/20.
//

#ifndef GRAPHICS_DEMO_TESTS_H
#define GRAPHICS_DEMO_TESTS_H

#include <vector>
#include <math.h>
#include "external/lodepng.h"

void testSavePng(const char* fileName) {
    int width = 400;
    int height = 400;
    int channels = 4;

    std::vector<unsigned char> image;
    image.reserve(width * height * channels);

    float a[] = { 0.5, 0.5, 0.5 };
    float b[] = { 0.5, 0.5, 0.5 };
    float c[] = { 1.0, 1.0, 1.0 };
    float d[] = { 0.00, 0.33, 0.67 };
    float color[] = { 0.0, 1.0, 0.0 };
    float mult = 255.0f;
    float w = 0.05f;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float t = (sin((float)i * w) - cos(w*(float)i + w*(float)j))/2.0f;
            for (int k = 0; k < 3; k++) {
                color[k] = a[k] + b[k] * cos(M_PI * (c[k] * t + d[k]));
            }
            image.push_back((unsigned char)(mult * color[0]));
            image.push_back((unsigned char)(mult * color[1]));
            image.push_back((unsigned char)(mult * color[2]));
            image.push_back((unsigned char)(mult * 1.0));
        }
    }

    //char nameBuffer[100];
    //sprintf(nameBuffer, fileName"/home/euclid/Experiments/graphics_demo/frames/test_%d.png", 0);
    unsigned error = lodepng::encode(fileName, image, width, height);
    if (error) printf("encoder error %d: %s", error, lodepng_error_text(error));
}

#endif //GRAPHICS_DEMO_TESTS_H
