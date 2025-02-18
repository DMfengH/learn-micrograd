#include "micrograd.h"

using Logger::info;
using Logger::warn;


void topoSort(ValuePtr root, std::vector<ValuePtr>& topo){
  Timer t("topoSort");
  std::set<ValuePtr> visited; 

  // topo排序 DFS逆后序法：
  // 后续遍历：保证子节点都访问过了才访问当前节点。最后在反过来，保证当前节点在所有子节点之前访问，即topo排序topo排序
  std::function<void(ValuePtr)> dfs = [&](ValuePtr root){
    if(visited.find(root) == visited.end()){
      visited.insert(root);
      for(ValuePtr p: root->prev_){
        dfs(p);
      }
      topo.push_back(root);
    }
  };

  // 总感觉这段代码太长，用了太多容器。
  // 入度法：
  std::function<void(ValuePtr)> bfs = [&](ValuePtr root){
    // 获取所有Node
    Timer* t1 = new Timer("get all Node ");
    std::unordered_set<ValuePtr> readyVisit;
    std::unordered_set<ValuePtr> visited;
    readyVisit.insert(root);

    while(!readyVisit.empty()){
      auto curIt = readyVisit.begin();
      ValuePtr cur = *(curIt);
      readyVisit.erase(cur);
      visited.insert(cur);

      for(const ValuePtr& prev: cur->prev_){
        if(visited.find(prev) == visited.end()){
          readyVisit.insert(prev);
        }
      }
    }
    delete t1;

    // 计算所有Node的outDegree
    Timer* t2 = new Timer("get outDegree");
    std::unordered_map<ValuePtr,int> outDegree;
    outDegree.reserve(visited.size()*2);
    for(const ValuePtr& vi: visited){
      for(const ValuePtr& pre: vi->prev_){
        outDegree[pre]++;
      }
    }
    delete t2;

    Timer* t3 = new Timer("get topo     ");
    // topo排序：outDegree为0，就加入到topo中。
    std::unordered_set<ValuePtr> outDegreeZero;
    outDegreeZero.insert(root);

    while(!outDegreeZero.empty()){
      auto curIt = outDegreeZero.begin();
      ValuePtr cur = *(curIt);  // 这里不能使用const引用，下面erase会把引用的对象删掉。
      outDegreeZero.erase(cur);
      topo.push_back(cur);
      for(const ValuePtr& prev: (cur)->prev_){
        outDegree[prev]--;
        if (outDegree[prev] == 0){
          outDegreeZero.insert(prev); // 这里也可以顺道把outDegree的元素erase
        }
      }
    }
    std::reverse(topo.begin(),topo.end());
    delete t3;
  };

  // dfs(root);
  bfs(root);
}

// 这里需要topo排序才行
void backward(ValuePtr root){
  Timer t("backward");
  std::vector<ValuePtr> topo;
  topoSort(root, topo);

  for (auto it=topo.rbegin(); it != topo.rend() ;it++){
    if((*it)->op.empty()){
      continue;
    }else{
      (*it)->backward_();
    } 
  }
}

void upgrade(ValuePtr root){
  std::deque<ValuePtr> readyVisit;
  std::vector<ValuePtr> visited;
  readyVisit.push_back(root);
  while(!readyVisit.empty()){
    ValuePtr cur = readyVisit.front();
    readyVisit.pop_front();
    visited.push_back(cur);
    for(ValuePtr prev: cur->prev_){
      if(std::find(visited.begin(), visited.end(), prev) == visited.end()){
        if(std::find(readyVisit.begin(), readyVisit.end(), prev) == readyVisit.end()){
          readyVisit.push_back(prev);
        }
      }
    }
  }

  info("visited size:", visited.size());
  for(size_t i=0; i < visited.size(); i++){
    visited[i]->val = visited[i]->val - 0.01*visited[i]->derivative; // 学习率
    visited[i]->derivative = 0;
  }
}

void updateParameters(MLP& mlp, double learningRate){
  std::unique_ptr<ValuePtr[]> parameters = mlp.parameters();
  size_t numParas=0;
  numParas += (mlp.inDegree+1)*mlp.outDegrees[0];
  for (size_t i=0; i< mlp.numLayers-1; i++){
      numParas += (mlp.outDegrees[i]+1)*mlp.outDegrees[i+1];
  }
  info("num of parameters in mlp:", numParas);
  for (size_t i=0; i<numParas; i++){
      parameters[i]->val += -1*learningRate*parameters[i]->derivative;
      parameters[i]->derivative = 0;
  }
}

std::vector<std::unique_ptr<ValuePtr[]>> computeOutput(MLP& mlp, const std::vector<std::unique_ptr<ValuePtr[]>>& inputs){
  std::vector<std::unique_ptr<ValuePtr[]>> yOut;
  for (int i=0; i<inputs.size(); i++){
    yOut.push_back(mlp(inputs[i]));
  }     

  return std::move(yOut);
}

// losses = [(1 + -yi*scorei).relu() for yi, scorei in zip(yb, scores)]
ValuePtr computePredictionLoss(const std::vector<std::unique_ptr<ValuePtr[]>>& yOut, const std::vector<ValuePtr>& yT){
  ValuePtr predictionLoss = std::make_shared<Value>();
  for (size_t i=0; i<yOut.size(); i++){
    predictionLoss = predictionLoss + relu((-(yT[i]*yOut[i][0]) + 1));
    // predictionLoss = predictionLoss + pow((yT[i]-yOut[i][0]),2); // 这里是有问题的，只用了最后一层的第一个节点作为输出和true_y比较
  }
  warn("check preLos: ", predictionLoss->val);
  predictionLoss = predictionLoss * pow(yT.size(),-1);
  warn("check preLos 2: ", predictionLoss->val);

  return predictionLoss;
}

// 想写成常量引用作为参数，但是mlp的函数没有实现const版，先这样
// parametersLoss，因为参数都很小，平方后乘以alpha就更小，所以导致ParametersLoss下降的很少
ValuePtr computeRegLoss(MLP& mlp, double alpha){
  ValuePtr regLoss = std::make_shared<Value>(0);
  std::unique_ptr<ValuePtr[]> parameters = mlp.parameters();

  size_t numParas=0;
  numParas += (mlp.inDegree+1)*mlp.outDegrees[0];
  for (size_t i=0; i< mlp.numLayers-1; i++){
      numParas += (mlp.outDegrees[i]+1)*mlp.outDegrees[i+1];
  }

  for (size_t i=0; i<numParas; i++){
      regLoss = regLoss + pow(parameters[i],2)*alpha;
  }

  return regLoss;
}

ValuePtr computeLoss(){
  ValuePtr totalLoss = std::make_shared<Value>();

}