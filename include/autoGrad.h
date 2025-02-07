#pragma once
#include "utilities.h"
#include "value.h"

#include <vector>
#include <algorithm>
#include <memory>
#include <set>
#include <deque>
#include <map>
#include <functional>


void topoSort(NodePtr root, std::vector<NodePtr>& topo);

void backward(NodePtr root);

