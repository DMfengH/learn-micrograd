#include "micrograd.h"

using Logger::info;
using Logger::warn;


void topoSort(ValuePtr root, std::vector<ValuePtr>& topo){
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

    // 计算所有Node的outDegree
    std::map<ValuePtr,int> outDegree;
    for(ValuePtr vi: visited){
      for(ValuePtr pre: vi->prev_){
        outDegree[pre]++;
      }
    }
    // topo排序：outDegree为0，就加入到topo中。
    std::vector<ValuePtr> outDegreeZero;
    outDegreeZero.push_back(root);

    while(!outDegreeZero.empty()){
      auto cur = *(outDegreeZero.begin());
      topo.insert(topo.begin(),cur);
      for(ValuePtr prev: (cur)->prev_){
        outDegree[prev]--;
        if (outDegree[prev] == 0){
          outDegreeZero.push_back(prev);
        }
      }
      outDegreeZero.erase(outDegreeZero.begin());
    }
  };

  // dfs(root);
  bfs(root);
}

// 这里需要topo排序才行
void backward(ValuePtr root){
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
