#include "data.h"

#include <cmath>

float lcg_uniform(unsigned* state) {
    *state = *state * 1664525u + 1013904223u;
    return (*state >> 8) / 16777216.0f;
}

float lcg_normal(unsigned* state) {
    const float u1 = lcg_uniform(state);
    const float u2 = lcg_uniform(state);
    return std::sqrt(-2.0f * std::log(u1 + 1e-7f)) * std::cos(6.2831853f * u2);
}

void make_spiral(float* features, int* labels, int points_per_class, int classes, float noise,
                 unsigned seed) {
    unsigned state = seed;
    for (int class_index = 0; class_index < classes; ++class_index) {
        for (int point = 0; point < points_per_class; ++point) {
            const float radius = static_cast<float>(point) / (points_per_class - 1);
            const float angle = 4.0f * class_index + 4.0f * radius +
                                noise * lcg_normal(&state);
            const int index = class_index * points_per_class + point;
            features[2 * index] = radius * std::sin(angle);
            features[2 * index + 1] = radius * std::cos(angle);
            labels[index] = class_index;
        }
    }
}
