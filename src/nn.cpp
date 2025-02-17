#include "nn.h"

std::mt19937 Neuron::gen(std::random_device{}());
std::uniform_real_distribution<double> Neuron::dist(-1.0, 1.0);