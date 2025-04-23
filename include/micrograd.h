#pragma once
#include <algorithm>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <stack>
#include <unordered_set>
#include <vector>

#include "engine.h"
#include "nn.h"
#include "utils.h"

void topoSort(ValuePtr root, std::vector<Value*>& topo);

void backward(ValuePtr root);

void updateParameters(MLP& mlp, double learningRate = 0.01);

void computeOutput(MLP& mlp, const std::vector<std::vector<InputVal>>& inputs,
                   std::vector<std::vector<ValuePtr>>& yOut);
void computeOutputBatchInput(const MLP& mlp, const std::vector<std::vector<InputVal>>& inputs,
                             std::vector<std::vector<ValuePtr>>& yOut);
void computeOutputSingleInput(const MLP& mlp, const std::vector<InputVal>& inputs,
                              std::vector<ValuePtr>& yOut);

// 这里的参数类型不太对，暂时先这样
ValuePtr computeLoss();

ValuePtr computePredictionLoss(const std::vector<std::vector<ValuePtr>>& yOut,
                               const std::vector<ValuePtr>& yT);
ValuePtr computePredictionLossSingleInput(const std::vector<ValuePtr>& yOut, const ValuePtr yT);
ValuePtr computeRegLoss(MLP& mlp);

void calculateGrad(std::vector<MLP>& mlps, MLP& mlp);