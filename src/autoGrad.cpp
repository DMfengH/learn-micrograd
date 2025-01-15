#include "autoGrad.h"

using Logger::info;
using Logger::warn;

static std::vector<NodePtr> alreadyVisited;

// 这里为什么使用智能指针？
// 因为Node的成员中有指向其他Node的指针 ×
// 因为要支持+num操作，而由num创建的Node对象，需要延长生命周期；使用shared_ptr让out共享所有权就比较好。
// 另外，捕获的内容会变化，所以只能是引用捕获，或者是值捕获指针,而引用捕获会遇到局部变量销毁的问题，所以最好用指针。
NodePtr operator+(NodePtr lhs, NodePtr rhs){
  NodePtr out = std::make_shared<Node>(lhs->val + rhs->val, '+', lhs, rhs);
  out->backward_ = [lhs, rhs, out](){     // 按照引用捕获，就会有问题！！！
      lhs->derivative += out->derivative;
      rhs->derivative += out->derivative;
    };
  return out;
}

NodePtr operator*(NodePtr lhs, NodePtr rhs){
  NodePtr out = std::make_shared<Node>(lhs->val * rhs->val, '*', lhs, rhs);
  out->backward_ = [lhs, rhs, out](){
      lhs->derivative += rhs->val * out->derivative;
      rhs->derivative += lhs->val * out->derivative;
    };
  return out;
}

// 这里这个num的所有权应该在哪里？？？最终是在它计算的out那里
// 为了把num传出去，只能使用智能指针
// 离开作用域还能用，就得是heap上的变量或者拷贝一份，方便管理所以用智能指针
NodePtr operator+(NodePtr lhs, int num){
  // static int numIndex = [](){info("First Visited"); return 0;}();
  // numIndex++;

  NodePtr numNode = std::make_shared<Node>(num, ' ', nullptr, nullptr);
  numNode->name = '/';
  return lhs + numNode;
}

void Node::buildTopo(std::vector<Node*>& topo, std::set<Node*>& visited){
  if(visited.count(this) == 0){
    topo.push_back(this);
    visited.insert(this);
    if(this->first){
      this->first->buildTopo(topo, visited);
    }
    if(this->second){
      this->second->buildTopo(topo, visited);
    }
  }
}

// 这里需要topo排序才行
void Node::backward(){
  std::vector<Node*> topo;
  std::set<Node*> visited;  // 使用NodePtr就会有问题，但并不是每次运行都有问题，并且调试运行就不会有问题。
  
  buildTopo(topo, visited);
  for(auto item:topo){
    if(item->op == ' '){
      continue;
    }else{
      item->backward_();
    } 
  }
}

Agnode_t* drawDataNode(NodePtr nodePtr, Agraph_t* g){
  // std::stringstream ss;
  // ss << nodePtr->name << " val:" << nodePtr->val << " grad:" << nodePtr->derivative;
  // std::string labelStr = ss.str();

  // char label[labelStr.size()+1];
  // strcpy(label, labelStr.c_str());
  // std::string st = "TTTom";
  Agnode_t* node = agnode(g, &(nodePtr->toString()[0]), TRUE);
  agsafeset(node, "shape", "box", "");
  agsafeset(node, "label",  &(nodePtr->toString()[0]), "");    
  return node;
}

Agnode_t* drawOpNode(NodePtr nodePtr, Agraph_t* g){
  std::stringstream ss;
  ss << nodePtr->first->name << nodePtr->op << nodePtr->second->name << nodePtr->name;
  std::string nameStr = ss.str();

  char name[nameStr.size()+1];
  strcpy(name, nameStr.c_str());

  Agnode_t* node = agnode(g, name, TRUE);
  agsafeset(node, "label", &nodePtr->op, "");
  return node;
}

void draw(NodePtr curNode, Agnode_t* curAgnode, Agraph_t* g){
  if (std::find(alreadyVisited.begin(), alreadyVisited.end(), curNode) != alreadyVisited.end()){
    return;
  }

  if (curNode->op == ' '){ return;}
  
  Agnode_t* opNode = drawOpNode(curNode, g);
  agedge(g, opNode, curAgnode, NULL, TRUE);
  alreadyVisited.push_back(curNode);

  if(curNode->first){
    Agnode_t* firstN = drawDataNode(curNode->first, g);
    agedge(g, firstN, opNode, NULL, TRUE);
    draw(curNode->first, firstN, g);
  }

  if(curNode->second){
    Agnode_t* secondN = drawDataNode(curNode->second, g);
    agedge(g, secondN, opNode, NULL, TRUE);
    draw(curNode->second, secondN, g);
  }
}

void drawGraph(NodePtr result, char* name, GVC_t* gvc){
  Agraph_t* graph = agopen(name, Agdirected, NULL);
  agsafeset(graph, "rankdir", "LR", "");    

  Agnode_t* resAgnode = drawDataNode(result, graph);
  draw(result, resAgnode, graph);
  alreadyVisited.clear();

  gvLayout(gvc, graph, "dot");
  gvRenderFilename(gvc, graph, "png", (std::string(name) + ".png").c_str());
}

void testAutoGrad(){
  NodePtr a = std::make_shared<Node>('a',5);
  NodePtr b = std::make_shared<Node>('b',2);
  NodePtr c = a+b;
  c->name = 'c';
  NodePtr d = std::make_shared<Node>('d',3);
  NodePtr e= c*d;
  e->name = 'e';
  NodePtr f = b * e;
  f->name = 'f';
  NodePtr g = f + 2;   // 这个+操作有问题, debug和release模式都有【已经修复】
  g->name = 'g';
  NodePtr h= g + e;
  h->name = 'h'; 
  h->derivative = 1;

  GVC_t* gvc = gvContext();
  drawGraph(h, "autograd" ,gvc);
  h->backward();
  drawGraph(h, "backward", gvc);
}