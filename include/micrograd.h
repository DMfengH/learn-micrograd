#pragma once
#include "utils.h"
#include "engine.h"
#include "nn.h"

#include <vector>
#include <algorithm>
#include <memory>
#include <set>
#include <unordered_set>
#include <deque>
#include <map>
#include <functional>


void topoSort(ValuePtr root, std::vector<ValuePtr>& topo);

void backward(ValuePtr root);

void updateParameters(MLP& mlp, double learningRate=0.01);

std::vector<std::unique_ptr<ValuePtr[]>> computeOutput(MLP& mlp, const std::vector<std::unique_ptr<ValuePtr[]>>& inputs);
// 这里的参数类型不太对，暂时先这样
ValuePtr computeLoss();

ValuePtr computePredictionLoss(const std::vector<std::unique_ptr<ValuePtr[]>>& yOut, const std::vector<ValuePtr>& yT);

ValuePtr computeRegLoss(MLP& mlp); 