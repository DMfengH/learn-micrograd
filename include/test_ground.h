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
#include <numeric>
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

void getInputsYT(std::vector<std::vector<InputVal>>& inputs, std::vector<ValuePtr>& yT) {
  std::ifstream file("../inputData.txt");
  if (!file) {
    warn("无法打开文件: ", "../inputData.txt");
    return;
  }
  double x, y, category;
  std::string line;
  while (std::getline(file, line)) {
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
  }
}

void testNN() {
  // Logger::setLogLevel(Logger::logLevel::WarnLevel);
  int outs[] = {16, 16, 1};  // 最后一层的维度，要和true_y的维度匹配。{16, 16, 1}
  MLP mlp(2, std::size(outs), outs);
  info("num of parameters in mlp:", mlp.parametersAll().size());

  std::vector<std::vector<InputVal>> inputs;
  std::vector<ValuePtr> yT;
  getInputsYT(inputs, yT);

  std::mt19937 batchGen(std::random_device{}());
  std::vector<int> numbers(100);
  std::iota(numbers.begin(), numbers.end(), 0);  // 生成 0~99 的数

  bool drawOrNot = true;
  std::ofstream lossForDrawF("../lossData.txt");

  double alpha = 0.001;  // 比0.1和0.01合适
  double learningRate = 1;
  int totalTime = 1000;
  int time = 0;
  while (time < totalTime) {
    std::unique_ptr<Timer> t = std::make_unique<Timer>("time cost per epoch");
    info("-------------------- epoch ", time, "--------------------------");
    ValuePtr predictionLoss;
    ValuePtr regLoss;
    ValuePtr totalLoss;

    std::vector<std::vector<InputVal>> batchInputs;
    std::vector<ValuePtr> batchYT;
    std::vector<std::vector<ValuePtr>> batchYOut;

    std::shuffle(numbers.begin(), numbers.end(), batchGen);
    for (size_t i = 0; i < 16; ++i) {
      int randomNum = numbers[i];
      batchInputs.push_back(inputs[randomNum]);
      batchYT.push_back(yT[randomNum]);
    }

    computeOutputBatchInput(mlp, batchInputs, batchYOut);

    predictionLoss = computePredictionLoss(batchYOut, batchYT);
    predictionLoss->modelPara = ModelPara::output;
    info("prediction_Loss: ", predictionLoss->val);

    regLoss = computeRegLoss(mlp);
    regLoss->modelPara = ModelPara::output;
    info("reg_Loss:", regLoss->val);

    totalLoss = predictionLoss + regLoss * alpha;
    // totalLoss = predictionLoss;
    info("totalLoss val:", totalLoss->val);
    lossForDrawF << time << " " << totalLoss->val << "\n";

    totalLoss->derivative = 1;
    backward(totalLoss);

    // if (time == 0) {
    //   GVC_t* gvc = gvContext();
    //   std::string name = "beforeUpdate";
    //   drawGraph(totalLoss, name, gvc);
    // }

    double decreaseLearningRate = learningRate - 0.9 * learningRate * time / totalTime;
    updateParameters(mlp, decreaseLearningRate);

    // if (time == 0) {
    //   GVC_t* gvc = gvContext();
    //   std::string name = "afterUpdate";
    //   drawGraph(totalLoss, name, gvc);
    // }
    info("num of Value: ", Value::maxID);
    info("num of ModelParas: ", mlp.numParametersW + mlp.numParametersB);

    time++;
    Value::cache.clear();
  }

  lossForDrawF.close();

  if (drawOrNot) {
    drawDivideGraph(mlp);
  }
}
