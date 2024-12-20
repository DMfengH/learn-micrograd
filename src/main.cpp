#include "utilities.h"

#include <graphviz/cgraph.h>
#include <graphviz/gvc.h>

#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <memory>
using Logger::info;


// 这个类如何在不使用shared_ptr的情况下实现operator+的闭包效果呢？？？？
class Node{
public:
  Node(int a_val): val(a_val){}
  Node(char* a_name): name(a_name){}
  Node(char* a_name, int a_val): name(a_name), val(a_val){}
  Node(int a_val, char* a_op, Node* a_first, Node* a_second): 
        val(a_val), op(a_op), first(a_first), second(a_second){}

  friend std::shared_ptr<Node> operator+(const std::shared_ptr<Node>& lhs, const std::shared_ptr<Node>& rhs);
  friend std::shared_ptr<Node> operator*(const std::shared_ptr<Node>& lhs, const std::shared_ptr<Node>& rhs);


public:
  int val=0;
  char* op = " ";
  Node* first = nullptr;
  Node* second = nullptr;
  float derivative = 0;
  char* name = " ";
  std::function<void()> backward_;
};

std::shared_ptr<Node> operator+(const std::shared_ptr<Node>& lhs, const std::shared_ptr<Node>& rhs){
  std::shared_ptr<Node> out = std::make_shared<Node>(lhs->val + rhs->val, "+", lhs, rhs);
  out->backward_ = [lhs, rhs, out](){
      lhs->derivative += out->derivative;
      rhs->derivative += out->derivative;
    };
  return out;
}

std::shared_ptr<Node> operator*(const std::shared_ptr<Node>& lhs, const std::shared_ptr<Node>& rhs){
  std::shared_ptr<Node> out = std::make_shared<Node>(lhs->val*rhs->val, "*", lhs, rhs);
  out->backward_ = [lhs, rhs, out](){
    lhs->derivative += rhs->val * out->derivative;
    rhs->derivative += lhs->val * out->derivative;
  };
  return out;
}

std::vector<Node*> alreadyVisited;

Agnode_t* drawNodeWithLable(Node* nodePtr, Agraph_t* g){
  std::stringstream ss;
  ss << nodePtr->name << " val:" << nodePtr->val << " grad:" << nodePtr->derivative;
  std::string labelStr = ss.str();

  char label[labelStr.size()+1];
  strcpy(label, labelStr.c_str());

  Agnode_t* node = agnode(g, nodePtr->name, TRUE);
  agsafeset(node, "label", label, "");    
  return node;
}

Agnode_t* drawOpNode(Node* nodePtr, Agraph_t* g){
  std::stringstream ss;
  ss << nodePtr->first->name << nodePtr->op << nodePtr->second->name << nodePtr->name;
  std::string nameStr = ss.str();

  char name[nameStr.size()+1];
  strcpy(name, nameStr.c_str());

  Agnode_t* node = agnode(g, name, TRUE);
  agsafeset(node, "shape", "box", "");
  agsafeset(node, "label", nodePtr->op, "");
  return node;
}

void plot(Node* curNode, Agnode_t* curAgnode, Agraph_t* g){
  if (std::find(alreadyVisited.begin(), alreadyVisited.end(), curNode) != alreadyVisited.end()){
    return;
  }

  if (curNode->op == " "){ return;}
  Agnode_t* opNode = drawOpNode(curNode, g);
  agedge(g, opNode, curAgnode, NULL, TRUE);
  alreadyVisited.push_back(curNode);

  if(curNode->first){
    Agnode_t* firstN = drawNodeWithLable(curNode->first, g);
    agedge(g, firstN, opNode, NULL, TRUE);
    plot(curNode->first, firstN, g);
  }

  if(curNode->second){
    Agnode_t* secondN = drawNodeWithLable(curNode->second, g);
    agedge(g, secondN, opNode, NULL, TRUE);
    plot(curNode->second, secondN, g);
  }
}

void plot(Node* resNode, Agraph_t* g){
  Agnode_t* resAgnode = drawNodeWithLable(resNode, g);

  plot(resNode, resAgnode, g);
  alreadyVisited.clear();
}

// 这里需要topo排序才行
void backward(Node* result){
  if (result == nullptr){
    return;
  }

  if(result->op == " "){
    return;
  }else{
    result->backward_();
  }

  backward(result->first);
  backward(result->second);
}

void drawGraph(Node* result, char* name, GVC_t* gvc){
  Agraph_t* graph = agopen(name, Agdirected, NULL);
  agsafeset(graph, "rankdir", "LR", "");    

  plot(result, graph);
  gvLayout(gvc, graph, "dot");
  gvRenderFilename(gvc, graph, "png", (std::string(name) + ".png").c_str());
}


// 返回一个捕获了外部变量的 Lambda
auto createLambda() {
    int x = 10;
    return [x]() { std::cout << "Captured x = " << x << std::endl; };
}


int main(){
  Node a{"a",5};
  Node b{"b",9};
  Node c=a+b;
  c.name = "c";
  Node d{"d",7};
  Node e= c*d;
  e.name = "e";
  Node f = b * e;
  f.name = "f";
  Node h= f + e;
  h.name = "h"; 


  // GVC_t* gvc = gvContext();
  // drawGraph(&h, "autograd" ,gvc);
  
  // backward(&h);
  // drawGraph(&h, "backward", gvc);


auto lambda = createLambda();  // 获取 Lambda 对象
lambda();  // 输出 "Captured x = 10"


  info("Graph has been generated ");
}