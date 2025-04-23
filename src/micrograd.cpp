#include "micrograd.h"
using Logger::info;
using Logger::warn;

void dfsStack(ValuePtr& root, std::vector<ValuePtr>& topo) {
  std::stack<std::pair<ValuePtr, bool>> st;
  std::unordered_set<ValuePtr> visited;
  st.push({root, false});

  while (!st.empty()) {
    auto [node, visitedFlag] = st.top();
    st.pop();

    if (visited.find(node) != visited.end()) continue;
    if (visitedFlag) {
      visited.insert(node);
      topo.push_back(node);
    } else {
      st.push({node, true});
      for (auto& p : node->prev_) {
        if (visited.find(p) == visited.end()) {
          st.push({p, false});
        }
      }
    }
  }
}

void dfsFunc(ValuePtr root, std::vector<ValuePtr>& topo,
             std::unordered_set<ValuePtr, ValuePtrHash, ValuePtrEqual>& visited) {
  for (ValuePtr& p : root->prev_) {
    if (visited.find(p) == visited.end()) {
      visited.insert(p);
      dfsFunc(p, topo, visited);
    }
  }
  topo.push_back(root);
}

void topoSort(ValuePtr root, std::vector<Value*>& topo) {
  /*
  set换成unorder_set 用时直接从40000降到8000ms
  for val 变成 for ref 用时降到6500ms
  使用Value*原始指针处理，会省一点时间,省500ms左右，
  把unordered_set换成std::vector<bool>，然后reserve空间，直接降到2000ms以下，但是要提前知道size.
  */

  // std::unordered_set<ValuePtr, ValuePtrHash, ValuePtrEqual> visited;  //
  // std::unordered_set<int> visited;  //
  std::vector<bool> visited(Value::maxID, false);  // maxID很大，不知道对后续影响大不大

  // topo排序 DFS逆后序法：
  // 后续遍历：保证子节点都访问过了才访问当前节点。最后在反过来，保证当前节点在所有子节点之前访问，即topo排序topo排序
  // std::function<void(ValuePtr&)> dfs = [&topo, &visited, &dfs](ValuePtr& root) {
  //   for (ValuePtr& p : root->prev_) {
  //     if (visited.find(p->id) == visited.end()) {
  //       visited.insert(p->id);
  //       dfs(p);
  //     }
  //   }
  //   topo.push_back(root);
  // };

  // 另一种不使用std::function的方法，理论上少一些开销。
  auto dfs = [](auto&& self, ValuePtr& root, std::vector<Value*>& topo,
                std::vector<bool>& visited) -> void {
    for (ValuePtr& p : root->prev_) {
      // if (visited.find(p->id) == visited.end()) {
      //   visited.insert(p->id);
      //   self(self, p, topo, visited);
      // }
      if (visited[p->id] == false) {
        visited[p->id] = true;
        self(self, p, topo, visited);
      }
    }
    topo.push_back(root.get());
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
    // std::unordered_set<ValuePtr> outDegreeZero;
    // outDegreeZero.insert(root);

    // while (!outDegreeZero.empty()) {
    //   auto curIt = outDegreeZero.begin();
    //   ValuePtr cur = *(curIt);  // 这里不能使用const引用，下面erase会把引用的对象删掉。
    //   outDegreeZero.erase(cur);
    //   topo.push_back(cur);
    //   for (const ValuePtr& prev : (cur)->prev_) {
    //     outDegree[prev]--;
    //     if (outDegree[prev] == 0) {
    //       outDegreeZero.insert(prev);  // 这里也可以顺道把outDegree的元素erase
    //     }
    //   }
    // }
    // std::reverse(topo.begin(), topo.end());
    // delete t3;
  };

  // dfsStack(root, topo);  // 没有下面的dfs效率高
  // dfsFunc(root, topo, visited);   // 没有下面的dfs效率高
  dfs(dfs, root, topo, visited);

  // bfs(root);
}

// 这里需要topo排序才行
void backward(ValuePtr root) {
  // Timer t("backward");
  std::vector<Value*> topo;  // 如果model不变化，这个topo也是不变的，不需要再次topoSort()

  topoSort(root, topo);

  // for (auto it = topo.rbegin(); it != topo.rend(); it++) {
  //   if ((*it)->op == Operation::INVALID) {
  //     continue;
  //   } else {
  //     (*it)->backward();
  //   }
  // }
  for (int i = topo.size() - 1; i >= 0; --i) {
    if ((topo[i])->op == Operation::INVALID) {
      continue;
    } else {
      (topo[i])->backward();
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
  std::vector<ValuePtr> parameters = mlp.parametersAll();
  size_t numParas = mlp.numParametersW + mlp.numParametersB;

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
void computeOutputBatchInput(const MLP& mlp, const std::vector<std::vector<InputVal>>& inputs,
                             std::vector<std::vector<ValuePtr>>& yOut) {
  yOut.clear();
  yOut.reserve(inputs.size());
  for (int i = 0; i < inputs.size(); i++) {
    yOut.push_back(mlp(inputs[i]));
  }
}

void computeOutputSingleInput(const MLP& mlp, const std::vector<InputVal>& inputs,
                              std::vector<ValuePtr>& yOut) {
  yOut = mlp(inputs);
}

// losses = [(1 + -yi*scorei).relu() for yi, scorei in zip(yb, scores)]
// 这里调用relu是为了简单实现max(0,x)函数，而不是调用激活函数，所以这里relu不能改成leakyRelu。
ValuePtr computePredictionLoss(const std::vector<std::vector<ValuePtr>>& yOut,
                               const std::vector<ValuePtr>& yT) {
  ValuePtr predictionLoss = hingeLoss(yOut, yT);

  // 铰链损失比均方误差的训练效果好，为什么呢？ 【relu是铰链损失的一部分】
  // 铰链损失是为分类问题设计的，它直接优化分类边界。鼓励模型在正确的分类上给出高置信度（即远离决策边界），这对于分类问题是更直接的优化目标。
  // 均方误差适用于回归问题，但在这里被用于分类，它会尝试使预测值尽可能接近真实标签，但没有考虑分类边界的最大化。
  // predictionLoss = predictionLoss + relu((-(yT[i] * yOut[i][0]) + 1));
  // predictionLoss = predictionLoss + pow((yT[i]-yOut[i][0]),2);
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
  std::vector<ValuePtr> parameters = mlp.parametersAll();

  // 大多数深度学习框架（如PyTorch、TensorFlow）在实现正则化时，
  // 直接计算所有参数的平方和，不会除以参数个数。这是因为λ,已经通过超参数调优控制了正则化的强度。
  ValuePtr regL = regLoss(parameters);

  return regL;
}

// ValuePtr computeLoss() { ValuePtr totalLoss = std::make_shared<Value>(); }

void calculateGrad(std::vector<MLP>& mlps, MLP& mlp) {
  std::vector<ValuePtr> Paras = mlp.parametersAll();
  std::vector<std::vector<ValuePtr>> allParas;

  // 这里出问题了，没有写引用，都是拷贝，拷贝构造会重新构造底层的W和b,置零。
  for (auto& m : mlps) {
    allParas.push_back(m.parametersAll());
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
