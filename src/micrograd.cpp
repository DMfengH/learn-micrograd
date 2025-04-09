#include "micrograd.h"
using Logger::info;
using Logger::warn;

void topoSort(ValuePtr root, std::vector<ValuePtr>& topo) {
  // Timer t("topoSort");
  std::set<ValuePtr> visited;

  // topo排序 DFS逆后序法：
  // 后续遍历：保证子节点都访问过了才访问当前节点。最后在反过来，保证当前节点在所有子节点之前访问，即topo排序topo排序
  std::function<void(ValuePtr)> dfs = [&](ValuePtr root) {
    if (visited.find(root) == visited.end()) {
      visited.insert(root);
      for (ValuePtr p : root->prev_) {
        dfs(p);
      }
      topo.push_back(root);
    }
  };

  // 总感觉这段代码太长，用了太多容器。
  // 入度法：
  std::function<void(ValuePtr)> bfs = [&](ValuePtr root) {
    // 获取所有Node
    // Timer* t1 = new Timer("get all Node ");
    std::unordered_set<ValuePtr> readyVisit;
    std::unordered_set<ValuePtr> visited;
    readyVisit.insert(root);

    while (!readyVisit.empty()) {
      auto curIt = readyVisit.begin();
      ValuePtr cur = *(curIt);
      readyVisit.erase(cur);
      visited.insert(cur);

      for (const ValuePtr& prev : cur->prev_) {
        if (visited.find(prev) == visited.end()) {
          readyVisit.insert(prev);
        }
      }
    }
    // delete t1;

    // 计算所有Node的outDegree
    // Timer* t2 = new Timer("get outDegree");
    std::unordered_map<ValuePtr, int> outDegree;
    outDegree.reserve(visited.size() * 2);
    for (const ValuePtr& vi : visited) {
      for (const ValuePtr& pre : vi->prev_) {
        outDegree[pre]++;
      }
    }
    // delete t2;
    // Timer* t3 = new Timer("get topo     ");
    // topo排序：outDegree为0，就加入到topo中。
    std::unordered_set<ValuePtr> outDegreeZero;
    outDegreeZero.insert(root);

    while (!outDegreeZero.empty()) {
      auto curIt = outDegreeZero.begin();
      ValuePtr cur = *(curIt);  // 这里不能使用const引用，下面erase会把引用的对象删掉。
      outDegreeZero.erase(cur);
      topo.push_back(cur);
      for (const ValuePtr& prev : (cur)->prev_) {
        outDegree[prev]--;
        if (outDegree[prev] == 0) {
          outDegreeZero.insert(prev);  // 这里也可以顺道把outDegree的元素erase
        }
      }
    }
    std::reverse(topo.begin(), topo.end());
    // delete t3;
  };

  // dfs(root);
  bfs(root);
}

// 这里需要topo排序才行
void backward(ValuePtr root) {
  // Timer t("backward");
  std::vector<ValuePtr> topo;  // 如果model不变化，这个topo也是不变的，不需要再次topoSort()
  topoSort(root, topo);

  for (auto it = topo.rbegin(); it != topo.rend(); it++) {
    if ((*it)->op == Operation::INVALID) {
      continue;
    } else {
      (*it)->backward();
    }
  }
}

void upgrade(ValuePtr root) {
  std::deque<ValuePtr> readyVisit;
  std::vector<ValuePtr> visited;
  readyVisit.push_back(root);
  while (!readyVisit.empty()) {
    ValuePtr cur = readyVisit.front();
    readyVisit.pop_front();
    visited.push_back(cur);
    for (ValuePtr prev : cur->prev_) {
      if (std::find(visited.begin(), visited.end(), prev) == visited.end()) {
        if (std::find(readyVisit.begin(), readyVisit.end(), prev) == readyVisit.end()) {
          readyVisit.push_back(prev);
        }
      }
    }
  }

  info("visited size:", visited.size());
  for (size_t i = 0; i < visited.size(); i++) {
    visited[i]->val = visited[i]->val - 0.01 * visited[i]->derivative;  // 学习率
    visited[i]->derivative = 0;
  }
}

void updateParameters(MLP& mlp, double learningRate) {
  std::vector<ValuePtr> parameters = mlp.parameters();
  size_t numParas = 0;
  numParas += (mlp.inDegree + 1) * mlp.outDegrees[0];
  for (size_t i = 0; i < mlp.numLayers - 1; i++) {
    numParas += (mlp.outDegrees[i] + 1) * mlp.outDegrees[i + 1];
  }

  for (size_t i = 0; i < numParas; i++) {
    parameters[i]->val += -1 * learningRate * parameters[i]->derivative;
    parameters[i]->derivative = 0;
  }
}

// 这里yOut直接就赋值成mlp的输出不行么？
void computeOutput(MLP& mlp, const std::vector<std::vector<InputVal>>& inputs,
                   std::vector<std::vector<ValuePtr>>& yOut) {
  std::vector<ValuePtr> tmp;
  if (yOut.empty()) {
    std::vector<ValuePtr> tmpY;
    for (int i = 0; i < inputs.size(); i++) {
      tmp = mlp(inputs[i]);
      for (auto x : tmp) {
        // 这里全都clone，最开始是为了多个输入有多个输出，否则都是同一个输出，但是现在不用了
        tmpY.push_back(x->clone());
      }
      yOut.push_back(tmpY);
      tmpY.clear();
    }
  } else {
    for (int i = 0; i < inputs.size(); i++) {
      tmp = mlp(inputs[i]);
      for (size_t j = 0; j < tmp.size(); ++j) {
        yOut[i][j]->val = tmp[j]->val;
        yOut[i][j]->derivative = 0;
      }
    }
  }
}
void computeOutputBatchInput(std::vector<MLP>& mlps,
                             const std::vector<std::vector<InputVal>>& inputs,
                             std::vector<std::vector<ValuePtr>>& yOut) {
  yOut.clear();
  for (int i = 0; i < inputs.size(); i++) {
    yOut.push_back(mlps[i](inputs[i]));
  }
}

void computeOutputSingleInput(MLP& mlp, const std::vector<InputVal>& inputs,
                              std::vector<ValuePtr>& yOut) {
  yOut = mlp(inputs);
}

// losses = [(1 + -yi*scorei).relu() for yi, scorei in zip(yb, scores)]
// 这里调用relu是为了简单实现max(0,x)函数，而不是调用激活函数，所以这里relu不能改成leakyRelu。
ValuePtr computePredictionLoss(const std::vector<std::vector<ValuePtr>>& yOut,
                               const std::vector<ValuePtr>& yT) {
  // 这里如果创建新的Value，所以后续计算都会产生新的Value，而不是在旧的上计算
  ValuePtr predictionLoss = relu((-(yT[0] * yOut[0][0]) + 1));

  for (size_t i = 1; i < yOut.size(); i++) {
    // 铰链损失比均方误差的训练效果好，为什么呢？ 【relu是铰链损失的一部分】
    // 铰链损失是为分类问题设计的，它直接优化分类边界。鼓励模型在正确的分类上给出高置信度（即远离决策边界），这对于分类问题是更直接的优化目标。
    // 均方误差适用于回归问题，但在这里被用于分类，它会尝试使预测值尽可能接近真实标签，但没有考虑分类边界的最大化。
    predictionLoss = predictionLoss + relu((-(yT[i] * yOut[i][0]) + 1));
    // predictionLoss = predictionLoss + pow((yT[i]-yOut[i][0]),2); //
    // 这里是有问题的，只用了最后一层的第一个节点作为输出和true_y比较
  }

  // predictionLoss = pow(predictionLoss, 0.5);
  // predictionLoss = predictionLoss * pow(yT.size(), -1);

  return predictionLoss;
}

ValuePtr computePredictionLossSingleInput(const std::vector<ValuePtr>& yOut, const ValuePtr yT) {
  // 这里如果创建新的Value，所以后续计算都会产生新的Value，而不是在旧的上计算
  ValuePtr predictionLoss = relu((-(yT * yOut[0]) + 1));
  return predictionLoss;
}

// 想写成常量引用作为参数，但是mlp的函数没有实现const版，先这样
// parametersLoss，因为参数都很小，导致ParametersLoss下降的很少??
ValuePtr computeRegLoss(MLP& mlp) {
  std::vector<ValuePtr> parameters = mlp.parameters();
  size_t numParas = 0;
  numParas += (mlp.inDegree + 1) * mlp.outDegrees[0];
  for (size_t i = 0; i < mlp.numLayers - 1; i++) {
    numParas += (mlp.outDegrees[i] + 1) * mlp.outDegrees[i + 1];
  }

  // 为什么不创建一个空白的regLoss然后在循环里从i=0开始累加？
  // 因为这样可以实现Value的重复使用，不会因为创建一个新的Value，导致后面所有的Value都需要重新创建
  // 现在没用了
  ValuePtr regLoss = pow(parameters[0], 2);
  for (size_t i = 1; i < numParas; i++) {
    regLoss = regLoss + pow(parameters[i], 2);
  }

  return regLoss;
}

ValuePtr computeLoss() { ValuePtr totalLoss = std::make_shared<Value>(); }

void calculateGrad(std::vector<MLP>& mlps, MLP& mlp) {
  std::vector<ValuePtr> Paras = mlp.parameters();
  std::vector<std::vector<ValuePtr>> allParas;

  // 这里出问题了，没有写引用，都是拷贝，拷贝构造会重新构造底层的W和b,置零。
  for (auto& m : mlps) {
    allParas.push_back(m.parameters());
  }
  int batchSize = allParas.size();
  for (size_t i = 0; i < Paras.size(); ++i) {
    double deri = 0;
    for (size_t j = 0; j < batchSize; ++j) {
      deri += allParas[j][i]->derivative;
    }
    // 为什么是+=？ 因为mlp在computeRegLoss中参与了计算，反传的时候有derivative。
    Paras[i]->derivative += deri / batchSize;

    // info("derivate val:", Paras[i]->derivative);
  }
}
