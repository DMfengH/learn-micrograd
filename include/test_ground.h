#pragma once
#include <graphviz/cgraph.h>
#include <graphviz/gvc.h>

#include <algorithm>
#include <bitset>
#include <fstream>
#include <functional>
#include <iomanip>  // 用于格式化输出，例如： setprecision 和 fixed
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "micrograd.h"
#include "nn.h"
#include "utils.h"
#include "visualize_tool.h"
// unistd是调用Unix系统API的；limits包含了一系列宏，表示了不同数据类型能表示的最大值和最小值，例如INT_MAX
#include <limits.h>
#include <unistd.h>

using Logger::info;
using Logger::warn;

#define BIT(x) (1 << x)

void test() {
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
  drawGraph(result, "Node_p", gvc);
  backward(result);
  drawGraph(result, "Node_b", gvc);
}

void testReadTxt() {
  std::ifstream file("../inputData.txt");
  if (!file) {
    warn("无法打开文件");
    return;
  }
  double x, y, category;
  std::string line;
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    if (iss >> x >> y >> category) {
      info(x, "-", y, "-", category, "\n");
    } else {
      warn("格式错误");
    }
  }
}

void testNN() {
  // Logger::setLogLevel(Logger::logLevel::WarnLevel);
  int outs[] = {16, 16, 1};  // 最后一层的维度，要和true_y的维度匹配。
  MLP mlp(2, std::size(outs), outs);

  std::vector<std::vector<ValuePtr>> inputs;
  std::vector<ValuePtr> yT;

  std::ifstream file("../inputData.txt");
  if (!file) {
    warn("无法打开文件: ", "../inputData.txt");
    return;
  }
  size_t numInput = 99;
  double x, y, category;
  std::string line;
  while (std::getline(file, line) && numInput) {
    std::istringstream iss(line);
    if (iss >> x >> y >> category) {
      std::vector<ValuePtr> input(2);
      input[0] = std::make_shared<Value>(x);
      input[1] = std::make_shared<Value>(y);

      inputs.push_back(std::move(input));
      yT.push_back(std::make_shared<Value>(category));
    } else {
      warn("格式错误");
    }
    numInput--;
  }

  bool drawOrNot = true;

  std::vector<std::vector<ValuePtr>> yOut;
  ValuePtr predictionLoss = std::make_shared<Value>();
  ValuePtr regLoss;
  ValuePtr totalLoss;

  double alpha = 0.001;  // 比0.1和0.01合适
  double learningRate = 1.0;
  int totalTime = 100;
  int time = 0;
  while (time < totalTime) {
    std::unique_ptr<Timer> t = std::make_unique<Timer>("time cost per epoch");
    info("-------------------- epoch ", time, "--------------------------");

    computeOutput(mlp, inputs, yOut);

    predictionLoss = computePredictionLoss(yOut, yT);
    info("prediction_Loss: ", predictionLoss->val);

    regLoss = computeRegLoss(mlp);
    info("reg_Loss:", regLoss->val);

    totalLoss = predictionLoss + regLoss * alpha;
    info("totalLoss val:", totalLoss->val);

    // 在这段代码输出的图像中，如何找哪些是w和b？在update()之后，w和b梯度会置零。
    // GVC_t* gvc = gvContext();
    // std::string name1 = "before";
    // drawGraph(loss, name1 ,gvc);

    totalLoss->derivative = 1;
    backward(totalLoss);

    // std::string name3 = "middle";
    // drawGraph(loss, name3, gvc);

    learningRate = 1 - 0.9 * time / totalTime;
    updateParameters(mlp, learningRate);

    // std::string name2 = "after";
    // drawGraph(loss, name2, gvc);
    time = time + 1;
  }

  auto start = std::chrono::high_resolution_clock::now();
  if (drawOrNot) {
    std::unique_ptr<Timer> td = std::make_unique<Timer>("time cost for draw");
    std::vector<std::vector<ValuePtr>> inputsForDraw;
    inputsForDraw.reserve(3600);
    // std::vector<ValuePtr> input;
    // input.reserve(2);

    // for (int iStep = 0; iStep < 60; ++iStep) {
    //   double i = (iStep - 30) * 0.1;
    //   for (int jStep = 0; jStep < 60; ++jStep) {
    //     double j = (jStep - 30) * 0.1;
    //     input.clear();
    //     input.emplace_back(std::make_shared<Value>(i));
    //     input.emplace_back(std::make_shared<Value>(j));

    //     inputsForDraw.push_back(std::move(input));
    //   }
    // }

    std::mutex mtx;
    auto generateChunk = [&inputsForDraw, &mtx](int iStart, int iEnd) {
      std::vector<std::vector<ValuePtr>> localInputsForDraw;
      localInputsForDraw.reserve((iEnd - iStart) * 60);
      std::vector<ValuePtr> input;
      input.reserve(2);

      for (int iStep = iStart; iStep < iEnd; ++iStep) {
        double i = (iStep - 30) * 0.1;
        for (int jStep = 0; jStep < 60; ++jStep) {
          double j = (jStep - 30) * 0.1;
          input.clear();
          input.emplace_back(std::make_shared<Value>(i));
          input.emplace_back(std::make_shared<Value>(j));

          localInputsForDraw.push_back(std::move(input));
        }
      }
      std::lock_guard<std::mutex> lock(mtx);
      inputsForDraw.insert(inputsForDraw.end(), localInputsForDraw.begin(),
                           localInputsForDraw.end());
    };
    std::thread t1(generateChunk, 0, 30);
    generateChunk(30, 60);
    t1.join();

    Timer* loop = new Timer("time cost in loop");
    std::vector<std::vector<ValuePtr>> inputsForDraw1;
    std::vector<std::vector<ValuePtr>> inputsForDraw2;
    inputsForDraw1.reserve(inputsForDraw.size() / 2);
    inputsForDraw2.reserve(inputsForDraw.size() / 2);
    inputsForDraw1.insert(inputsForDraw1.end(), inputsForDraw.begin(),
                          inputsForDraw.begin() + inputsForDraw.size() / 2);
    inputsForDraw2.insert(inputsForDraw2.end(), inputsForDraw.begin() + inputsForDraw.size() / 2,
                          inputsForDraw.end());
    std::vector<std::vector<ValuePtr>> yForDraw1;
    std::vector<std::vector<ValuePtr>> yForDraw2;

    std::thread t3(computeOutput, std::ref(mlp), std::ref(inputsForDraw1), std::ref(yForDraw1));

    std::thread t4(computeOutput, std::ref(mlp), std::ref(inputsForDraw2), std::ref(yForDraw2));

    t3.join();
    t4.join();

    std::vector<std::vector<ValuePtr>> yForDraw;
    yForDraw.reserve(inputsForDraw.size());
    yForDraw.insert(yForDraw.end(), yForDraw1.begin(), yForDraw1.end());
    yForDraw.insert(yForDraw.end(), yForDraw2.begin(), yForDraw2.end());
    delete loop;

    inputsForDraw1.clear();
    inputsForDraw2.clear();
    yForDraw1.clear();
    yForDraw2.clear();

    info("inputsize: ", inputsForDraw.size(), "  outsize: ", yForDraw.size());
    std::ofstream fileOut("../outPutForDraw.txt");

    fileOut << std::fixed << std::setprecision(2);
    for (size_t i = 0; i < inputsForDraw.size(); ++i) {
      fileOut << inputsForDraw[i][0]->val << " " << inputsForDraw[i][1]->val << " "
              << (yForDraw[i][0]->val > 0 ? 1 : 0) << "\n";
    }

    file.clear();
    file.seekg(0, std::ios::beg);
    fileOut << file.rdbuf();

    inputsForDraw.clear();

    start = std::chrono::high_resolution_clock::now();
    yForDraw.clear();
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "exit loacl scope took " << duration.count() << "ms\n";
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
  //     file << x << " " << (pow(e,(2*x))-1) / (pow(e,(2*x))+1) << " " << x <<
  //     "\n";
  // file.close();

  system(
      "gnuplot -e \"set palette defined (0 'red', 1 'blue', 2 'green'); set "
      "xrange [-3:3]; set yrange [-3:3]; plot '../data.txt' with points pt 7 "
      "ps 1.5 palette notitle; pause -1\"");
  return 0;
}

void testAutoGrad() {
  ValuePtr a = std::make_shared<Value>(5);
  ValuePtr b = std::make_shared<Value>(2);
  ValuePtr c = a + b;
  ValuePtr d = std::make_shared<Value>(3);

  ValuePtr e = c * d;
  ValuePtr f = b * e;
  //   ValuePtr g = f + 20;   // 这个+操作有问题,
  //   debug和release模式都有【已经修复】

  ValuePtr h = f + e;

  h->derivative = 1;

  GVC_t* gvc = gvContext();
  drawGraph(h, "../autograd", gvc);
  backward(h);
  drawGraph(h, "../backward", gvc);
}

void testCache() {
  {
    ValuePtr a = std::make_shared<Value>(5);
    ValuePtr b = std::make_shared<Value>(2);
    ValuePtr c = a + b;
    c = a + b;
  }
}
