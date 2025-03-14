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

#define BIT(x) (1 << x) t

void testNN() {
  // Logger::setLogLevel(Logger::logLevel::WarnLevel);
  int outs[] = {16, 16, 1};  // 最后一层的维度，要和true_y的维度匹配。{16, 16, 1}
  MLP mlp(2, std::size(outs), outs);

  std::vector<std::vector<InputVal>> inputs;
  std::vector<ValuePtr> yT;

  std::ifstream file("../inputData.txt");
  if (!file) {
    warn("无法打开文件: ", "../inputData.txt");
    return;
  }
  size_t numInput = 100;
  double x, y, category;
  std::string line;
  while (std::getline(file, line) && numInput) {
    std::istringstream iss(line);
    if (iss >> x >> y >> category) {
      std::vector<InputVal> input;
      input.emplace_back(x);
      input.emplace_back(y);

      inputs.push_back(input);
      yT.push_back(std::make_shared<Value>(category));
    } else {
      warn("格式错误");
    }
    numInput--;
  }

  bool drawOrNot = true;

  double alpha = 0.001;  // 比0.1和0.01合适
  double learningRate = 1.0;
  int totalTime = 100;
  int time = 0;
  while (time < totalTime) {
    std::unique_ptr<Timer> t = std::make_unique<Timer>("time cost per epoch");
    info("-------------------- epoch ", time, "--------------------------");
    double lossSum = 0;
    {
      std::vector<std::vector<ValuePtr>> yOut;

      ValuePtr predictionLoss;
      ValuePtr regLoss;
      ValuePtr totalLoss;

      std::vector<std::vector<InputVal>> batchInputs;
      std::vector<ValuePtr> batchYT;
      std::vector<std::vector<ValuePtr>> batchYOut;
      std::vector<MLP> mlps;

      for (size_t i = 0; i < 8; ++i) {
        int randomNum = std::rand() % 100;  // 不设置种子，就会获得相同的序列
        batchInputs.push_back(inputs[randomNum]);
        batchYT.push_back(yT[randomNum]);
        mlps.push_back(mlp);
      }

      computeOutputBatchInput(mlps, batchInputs, batchYOut);

      predictionLoss = computePredictionLoss(batchYOut, batchYT);
      info("prediction_Loss: ", predictionLoss->val);

      // GVC_t* gvc = gvContext();
      // std::string name1 = "before";
      // drawGraph(predictionLoss, name1, gvc);

      // regLoss = computeRegLoss(mlp);
      // info("reg_Loss:", regLoss->val);

      // totalLoss = predictionLoss + regLoss * alpha;
      totalLoss = predictionLoss;
      info("totalLoss val:", totalLoss->val);

      totalLoss->derivative = 1;
      backward(totalLoss);

      calculateGrad(mlps, mlp);

      learningRate = 1 - 0.9 * time / totalTime;
      updateParameters(mlp, learningRate);

      time = time + 1;
      Value::cache.clear();
      info("Value num: ", Value::maxID);
    }
  }

  auto start = std::chrono::high_resolution_clock::now();
  // /*
  if (drawOrNot) {
    std::unique_ptr<Timer> td = std::make_unique<Timer>("time cost for draw");
    std::vector<std::vector<InputVal>> inputsForDraw;
    inputsForDraw.reserve(3600);

    std::mutex mtx;
    auto generateChunk = [&inputsForDraw, &mtx](int iStart, int iEnd) {
      std::vector<std::vector<InputVal>> localInputsForDraw;
      localInputsForDraw.reserve((iEnd - iStart) * 60);
      std::vector<InputVal> input;
      input.reserve(2);

      for (int iStep = iStart; iStep < iEnd; ++iStep) {
        double i = (iStep - 30) * 0.1;
        for (int jStep = 0; jStep < 60; ++jStep) {
          double j = (jStep - 30) * 0.1;
          input.clear();
          input.emplace_back(i);
          input.emplace_back(j);

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
    std::vector<std::vector<InputVal>> inputsForDraw1;
    std::vector<std::vector<InputVal>> inputsForDraw2;
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
      fileOut << inputsForDraw[i][0].val << " " << inputsForDraw[i][1].val << " "
              << (yForDraw[i][0]->val > 0 ? 1 : 0) << "\n";
    }

    file.clear();
    file.seekg(0, std::ios::beg);
    fileOut << file.rdbuf();

    inputsForDraw.clear();

    start = std::chrono::high_resolution_clock::now();
    yForDraw.clear();
  }
  // */
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  std::cout << "exit loacl scope took " << duration.count() << "ms\n";
}
