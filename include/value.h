#pragma once
#include <memory>
#include <iostream>
#include <sstream>
#include <set>
#include <functional>

// 这个类如何在不使用shared_ptr的情况下实现operator+的闭包效果呢？？？？
class Node;
using NodePtr = std::shared_ptr<Node>;

class Node{
public:
  Node(float a_val): val(a_val), id(maxID++){}

  ~Node() {
        // std::cout << "Node destructed: " << std::endl;
    }

  friend NodePtr operator+(NodePtr lhs, NodePtr rhs);
  friend NodePtr operator*(NodePtr lhs, NodePtr rhs);
  friend NodePtr operator+(NodePtr lhs, float num);

  std::string toString(){
    std::stringstream ss;
    ss << "Value(data=" << this->val << ", grad=" << this->derivative << ")";
    return ss.str();
  }
  
private:
  static int maxID;

public:
  int id = 0;
  float val = 0;
  float derivative = 0;
  std::string op = "";
  std::set<NodePtr> prev_;
  std::function<void()> backward_ = nullptr;
  
};