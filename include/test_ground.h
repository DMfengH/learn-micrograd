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

void testReadTxt(){
    std::ifstream file("../data.txt");
    if(!file){
        warn("无法打开文件");
        return ;
    }
    double x, y, category;
    std::string line;
    while(std::getline(file, line)){
        std::istringstream iss(line);
        if(iss>> x >> y >> category){
            info(x,"-",y,"-",category,"\n");
        }else{
            warn("格式错误");
        }
    }
}

void testNN(){

    int outs[] = {16,16,1};   // 最后一层的维度，要和true_y的维度匹配。
    MLP mlp(2, std::size(outs), outs);

    std::vector<std::unique_ptr<ValuePtr[]>> inputs;
    std::vector<ValuePtr> ty;
    
    std::ifstream file("../data.txt");
    if(!file){
        warn("无法打开文件");
        return ;
    }
    size_t tim = 20;
    double x, y, category;
    std::string line;
    while(std::getline(file, line) && tim){
        std::istringstream iss(line);
        if(iss>> x >> y >> category){
            std::unique_ptr<ValuePtr[]> input0 = std::make_unique<ValuePtr[]>(2);
            input0[0] = std::make_shared<Value>(x);
            input0[1] = std::make_shared<Value>(y);
        
            inputs.push_back(std::move(input0));
            ty.push_back(std::make_shared<Value>(category));
            
        }else{
            warn("格式错误");
        }
        tim -=1;
        
    }


    int times =0;
    while(times <= 3){
        info("-------------------- epoch ", times , "--------------------------");
        std::vector<std::unique_ptr<ValuePtr[]>> anss;
        for (int i=0; i<inputs.size(); i++){
            anss.push_back(mlp(inputs[i]));
            info(anss[i][0]->val);
        }     
        info("anss size: ", anss.size());

        ValuePtr loss = std::make_shared<Value>();
        for (size_t i=0; i<ty.size(); i++){
            loss = loss + pow((ty[i]-anss[i][0]),2); // 这里是有问题的，只用了最后一层的第一个节点作为输出和true_y比较
        }

        info("loas val:", loss->val);
        
        loss->derivative = 1;
    
        // 在这段代码输出的图像中，如何找哪些是w和b？在update()之后，w和b梯度会置零。
        // GVC_t* gvc = gvContext();
        // std::string name1 = "before";
        // drawGraph(loss, name1 ,gvc);

        backward(loss);     // 这一步非常卡

        info("after backward");
        // std::string name3 = "middle";
        // drawGraph(loss, name3, gvc);

        update(mlp);

        // std::string name2 = "after";
        // drawGraph(loss, name2, gvc);
        times = times+1;
    }
    

}

// 真好用
int testGNUPlot() {
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) != nullptr) {
        std::cout << "Current working directory: " << cwd << std::endl;
    } else {
        std::cerr << "Error getting current working directory" << std::endl;
    }

    // double e = M_E;
    // std::ofstream file("data.txt");
    // for (double x = -10.0; x < 10.0; x += 0.1)
    //     file << x << " " << (pow(e,(2*x))-1) / (pow(e,(2*x))+1) << " " << x << "\n";
    // file.close();

    system("gnuplot -e \"set palette defined (0 'red', 1 'blue', 2 'green'); set xrange [-3:3]; set yrange [-3:3]; plot '../data.txt' with points pt 7 ps 1.5 palette notitle; pause -1\"");
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