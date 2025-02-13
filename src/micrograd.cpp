#include "autoGrad.h"

using Logger::info;
using Logger::warn;


void topoSort(NodePtr root, std::vector<NodePtr>& topo){
  std::set<NodePtr> visited; 

  // topo排序 DFS逆后序法：
  // 后续遍历：保证子节点都访问过了才访问当前节点。最后在反过来，保证当前节点在所有子节点之前访问，即topo排序topo排序
  std::function<void(NodePtr)> dfs = [&](NodePtr root){
    if(visited.find(root) == visited.end()){
      visited.insert(root);
      for(NodePtr p: root->prev_){
        dfs(p);
      }
      topo.push_back(root);
    }
  };

  // 总感觉这段代码太长，用了太多容器。
  // 入度法：
  std::function<void(NodePtr)> bfs = [&](NodePtr root){
    // 获取所有Node
    std::deque<NodePtr> readyVisit;
    std::vector<NodePtr> visited;
    readyVisit.push_back(root);
    while(!readyVisit.empty()){
      NodePtr cur = readyVisit.front();
      readyVisit.pop_front();
      visited.push_back(cur);
      for(NodePtr prev: cur->prev_){
        if(std::find(visited.begin(), visited.end(), prev) == visited.end()){
          if(std::find(readyVisit.begin(), readyVisit.end(), prev) == readyVisit.end()){
            readyVisit.push_back(prev);
          }
        }
      }
    }

    // 计算所有Node的inDegree
    std::map<NodePtr,int> inDegree;
    for(NodePtr node: visited){
      for(NodePtr pre: node->prev_){
        inDegree[pre]++;
      }
    }
    // topo排序：inDegree为0，就加入到topo中。
    std::vector<NodePtr> inDegreeZero;
    inDegreeZero.push_back(root);

    while(!inDegreeZero.empty()){
      auto cur = *(inDegreeZero.begin());
      topo.insert(topo.begin(),cur);
      for(NodePtr node: (cur)->prev_){
        inDegree[node]--;
        if (inDegree[node] ==0){
          inDegreeZero.push_back(node);
        }
      }
      inDegreeZero.erase(inDegreeZero.begin());
    }
  };

  // dfs(root);
  bfs(root);
}

// 这里需要topo排序才行
void backward(NodePtr root){
  std::vector<NodePtr> topo;
  topoSort(root, topo);

  for (auto it=topo.rbegin(); it != topo.rend() ;it++){
    if((*it)->op.empty()){
      continue;
    }else{
      (*it)->backward_();
    } 
  }
}


// void drawGraph2(NodePtr result, char* name, GVC_t* gvc){
//   std::vector<NodePtr> Nodes;
//   std::vector<std::pair<NodePtr, NodePtr>> edges;
//   std::vector<NodePtr> visited;
//   std::vector<NodePtr> waitForVisit;
//   waitForVisit.push_back(result);
//   while(!waitForVisit.empty()){

//   }
// }


