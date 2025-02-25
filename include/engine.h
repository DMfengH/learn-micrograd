#pragma once

#include "utils.h"
using Logger::info;
using Logger::warn;

#include <memory>
#include <iostream>
#include <sstream>
#include <set>
#include <functional>
#include <cmath>

// 这个类如何在不使用shared_ptr的情况下实现operator+的闭包效果呢？？？？
class Value;
using ValuePtr = std::shared_ptr<Value>;

struct PairHash {
  std::size_t operator()(const std::pair<ValuePtr, ValuePtr>& p) const {
      return std::hash<Value*>()(p.first.get()) ^ (std::hash<Value*>()(p.second.get()) << 1);
  }
};

// struct TupleHash {
//   std::size_t operator()(const std::tuple<ValuePtr, ValuePtr>& t) const {
//       return std::hash<Value*>()(std::get<0>(t).get()) ^ (std::hash<Value*>()(std::get<1>(t).get()) << 1);
//   }
// };

class Value{
public:
  Value():id(maxID++){
    std::cout << "construct a Value" << id << std::endl;
  }
  Value(double a_val): id(maxID++), val(a_val){
    std::cout << "construct a Value++" << id << std::endl;
  }

  Value(const Value& other): id(maxID++), 
                              val(other.val), 
                              derivative(other.derivative),
                              op(other.op),
                              prev_(other.prev_),
                              backward_(backward_){
    info("copy construct called");
  }

  ~Value() {
    backward_ = nullptr;
    // std::cout<< "destory a Value: " << id << std::endl; 
  }

  // karpathy的python实现中，把backward作为一个成员【和pytorch的API是一致的】，
  // 这个版本把backward单独作为一个全局函数，所以就不实现这个backward成员函数。
  // void backward(); 

  // 所有的这些计算操作，都是创建新的res，而不是修改res值
  friend ValuePtr operator+(ValuePtr lhs, ValuePtr rhs);
  friend ValuePtr operator*(ValuePtr lhs, ValuePtr rhs);
  friend ValuePtr operator+(ValuePtr lhs, double num);
  friend ValuePtr operator*(ValuePtr lhs, double num);

  friend ValuePtr operator-(ValuePtr lhs, ValuePtr rhs); 
  friend ValuePtr operator/(ValuePtr lhs, ValuePtr rhs);
  
  friend ValuePtr operator-(ValuePtr vp);
  friend ValuePtr inv(ValuePtr vp);
  friend ValuePtr exp(ValuePtr vp); 
  friend ValuePtr pow(ValuePtr lhs, double num);
  friend ValuePtr pow(ValuePtr lhs, ValuePtr rhs);
  friend ValuePtr tanh(ValuePtr vp);
  friend ValuePtr relu(ValuePtr vp);

  std::string toString(){
    std::stringstream ss;
    ss << "data=" << this->val << " | grad=" << this->derivative;
    return ss.str();
  }
  
private:
  static int maxID; // 当创建的Value很多的时候，这个ID可能超过最大值
  static std::unordered_map<std::pair<ValuePtr, ValuePtr>, ValuePtr, PairHash> cache;
  static ValuePtr placeHolder;

public:
  int id = 0;
  double val = 0;
  double derivative = 0;
  std::string op = "";
  std::set<ValuePtr> prev_;
  std::function<void()> backward_ = nullptr;
  
};