#pragma once
#include "utils.h"
#include "engine.h"

#include <vector>
#include <algorithm>
#include <memory>
#include <set>
#include <deque>
#include <map>
#include <functional>


void topoSort(ValuePtr root, std::vector<ValuePtr>& topo);

void backward(ValuePtr root);

