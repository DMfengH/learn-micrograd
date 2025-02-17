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

void update(MLP& mlp);