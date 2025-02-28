#pragma once
#include <algorithm>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <unordered_set>
#include <vector>

#include "engine.h"
#include "nn.h"
#include "utils.h"

void topoSort(ValuePtr root, std::vector<ValuePtr>& topo);

void backward(ValuePtr root);

void updateParameters(MLP& mlp, double learningRate = 0.01);

void computeOutput(MLP& mlp, const std::vector<std::vector<ValuePtr>>& inputs,
                   std::vector<std::vector<ValuePtr>>& yOut);
// 这里的参数类型不太对，暂时先这样
ValuePtr computeLoss();

ValuePtr computePredictionLoss(const std::vector<std::vector<ValuePtr>>& yOut,
                               const std::vector<ValuePtr>& yT);

ValuePtr computeRegLoss(MLP& mlp);