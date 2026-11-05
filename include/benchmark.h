#pragma once

#include "mlp.h"

float time_train_step_ms(MLP& model, Workspace& workspace, const float* features,
                         const int* labels, int batch, float learning_rate, int repetitions);
void benchmark_training_steps();
