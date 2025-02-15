#pragma once
#include "utils.h"
#include "micrograd.h"
#include "visualize_tool.h"
#include "nn.h"

#include <graphviz/cgraph.h>
#include <graphviz/gvc.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <memory>
#include <set>
#include <functional>
#include <bitset>
// unistd是调用Unix系统API的；limits包含了一系列宏，表示了不同数据类型能表示的最大值和最小值，例如INT_MAX
#include <unistd.h>
#include <limits.h>

using Logger::info;
using Logger::warn;

#define BIT(x) (1<<x)

void test(){
    ValuePtr a = std::make_shared<Value>(2.0);
    ValuePtr b = std::make_shared<Value>(0);
    ValuePtr w1 = std::make_shared<Value>(-3.0);
    ValuePtr w2 = std::make_shared<Value>(1.0);
    ValuePtr aw1 = a * w1;
    ValuePtr bw2 = b * w2;
    ValuePtr aw1bw2 = aw1 + bw2;
    ValuePtr bias = std::make_shared<Value>(6.8813735);
    auto aw1bw2bias = aw1bw2 + bias;
    auto result = tanh(aw1bw2bias);

    result->derivative = 1;
    GVC_t* gvc = gvContext();
    drawGraph(result, "Node_p" ,gvc);
    backward(result);
    drawGraph(result, "Node_b", gvc);


}

void testNN(){

    Neuron n1(3);
    info("herer");
    Layer l1(4,1);
    info("herer1");
    int outs[] = {2,1};
    MLP mlp(2, std::size(outs), outs);
    std::unique_ptr<ValuePtr[]> input = std::make_unique<ValuePtr[]>(2);
    info("herer2");
    
    input[0] = std::make_shared<Value>(3);
    input[1] = std::make_shared<Value>(2);
    // input[2] = std::make_shared<Value>(2);
    // input[3] = std::make_shared<Value>(2);

    info("herer3");

    auto anss = mlp(input);
    ValuePtr ans = anss[0];
    info("herer4");
    
    ans->derivative = 1;
    GVC_t* gvc = gvContext();
    drawGraph(ans, "Node_p" ,gvc);
    backward(ans);
    drawGraph(ans, "Node_b", gvc);

}

// 真好用
int testGNUPlot() {
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) != nullptr) {
        std::cout << "Current working directory: " << cwd << std::endl;
    } else {
        std::cerr << "Error getting current working directory" << std::endl;
    }

    double e = M_E;
    std::ofstream file("data.txt");
    for (double x = -10.0; x < 10.0; x += 0.1)
        file << x << " " << (pow(e,(2*x))-1) / (pow(e,(2*x))+1) << " " << x << "\n";
    file.close();

    system("gnuplot -e \"set xrang [-10:10]; set yrange [-1.5:1.5]; splot 'data.txt' with lines; pause -1\"");
    return 0;
}

void testAutoGrad(){
  ValuePtr a = std::make_shared<Value>(5);
  ValuePtr b = std::make_shared<Value>(2);
  ValuePtr c = a+b;
  ValuePtr d = std::make_shared<Value>(3);

  ValuePtr e= c*d;
  ValuePtr f = b * e;
  ValuePtr g = f + 20;   // 这个+操作有问题, debug和release模式都有【已经修复】
  ValuePtr h= g + e;

  h->derivative = 1;

  GVC_t* gvc = gvContext();
  drawGraph(h, "autograd" ,gvc);
  backward(h);
  drawGraph(h, "backward", gvc);

  // verify the numerical derivative
  float ep = 0.001;
  // a->val = (a->val) + ep;
  // b->val = b->val +ep;
  // c = a+b;
  // c->val = c->val +ep; 
  d->val = d->val + ep;

  e = c*d;
  f = b*e;
  g = f+20;
  ValuePtr h2 = g+e;
  float deriv = (h2->val - h->val) / ep;
  info("derivative: ", deriv);

}