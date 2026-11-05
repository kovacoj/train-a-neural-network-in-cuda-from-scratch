#pragma once

float lcg_uniform(unsigned* state);
float lcg_normal(unsigned* state);
void make_spiral(float* features, int* labels, int points_per_class, int classes, float noise,
                 unsigned seed);
