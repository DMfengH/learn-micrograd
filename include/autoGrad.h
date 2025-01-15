#include "utilities.h"

#include <graphviz/cgraph.h>
#include <graphviz/gvc.h>

#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <memory>
#include <set>
#include <functional>
#include <bitset>

// 这个类如何在不使用shared_ptr的情况下实现operator+的闭包效果呢？？？？
class Node;
using NodePtr = std::shared_ptr<Node>;

class Node{
public:
  Node(int a_val): val(a_val){}
  Node(char a_name): name(a_name){}
  Node(char a_name, int a_val): name(a_name), val(a_val){}
  Node(int a_val, char a_op, NodePtr a_first, NodePtr a_second): 
        val(a_val), op(a_op), first(a_first), second(a_second){}

  ~Node() {
        // std::cout << "Node destructed: " << std::endl;
    }

  void backward();
  void buildTopo(std::vector<Node*>& topo, std::set<Node*>& visited);

  friend NodePtr operator+(NodePtr lhs, NodePtr rhs);
  friend NodePtr operator*(NodePtr lhs, NodePtr rhs);
  friend NodePtr operator+(NodePtr lhs, int num);

  std::string toString(){
    std::stringstream ss;
    ss << "Value(data=" << this->val << ", grad=" << this->derivative << ")";
    return ss.str();
  }
  

public:
  int val=0;
  char op = ' ';
  NodePtr first = nullptr;
  NodePtr second = nullptr;
  float derivative = 0;
  char name = ' ';
  std::function<void()> backward_ = nullptr;
};




void testAutoGrad();

