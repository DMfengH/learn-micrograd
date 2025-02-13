#pragma once
#include "utils.h"
#include "micrograd.h"
#include "visualize_tool.h"

#include <graphviz/cgraph.h>
#include <graphviz/gvc.h>

#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <memory>
#include <set>
#include <functional>
#include <bitset>

using Logger::info;
using Logger::warn;

#define BIT(x) (1<<x)


void testAutoGrad(){
  NodePtr a = std::make_shared<Node>(5);
  NodePtr b = std::make_shared<Node>(2);
  NodePtr c = a+b;
  NodePtr d = std::make_shared<Node>(3);

  NodePtr e= c*d;
  NodePtr f = b * e;
  NodePtr g = f + 20;   // 这个+操作有问题, debug和release模式都有【已经修复】
  NodePtr h= g + e;

  h->derivative = 1;

  GVC_t* gvc = gvContext();
  drawGraph(h, "autograd" ,gvc);
  backward(h);
  drawGraph(h, "backward", gvc);
}