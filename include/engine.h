#pragma once

#include "utils.h"
using Logger::error;
using Logger::info;
using Logger::warn;

#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <set>
#include <sstream>

enum class Operation {
  INVALID,
  ADD,
  SUBTRACT,
  MULTIPLY,
  DIVIDE,
  TANH,
  NEG,
  INV,
  EXP,
  LOG,
  POW,
  RELU,
  MATMUL,

  SUM,
  AVG,

  ADDI,
  MULI,
  POWI,
  MATMULI,

  HINGELOSS,
  REGLOSS,
  LOGITLOSS,

  WMULXADDB,
  WMULXADDBI,

};

enum class TrainOrEval {
  train,
  eval,
};

enum class ModelPara {
  notModelPara,
  ParaW,
  Parab,
  output,
  input,
};
std::string toString(Operation op);

// 这个类如何在不使用shared_ptr的情况下实现operator+的闭包效果呢？？？？
class InputVal;
class Value;
using ValuePtr = std::shared_ptr<Value>;

struct PairHash {
  std::size_t operator()(const std::pair<ValuePtr, ValuePtr>& p) const {
    return std::hash<Value*>()(p.first.get()) ^ (std::hash<Value*>()(p.second.get()) << 1);
  }
};

// struct TupleHash {
//   std::size_t operator()(const std::tuple<ValuePtr, ValuePtr>& t) const {
//       return std::hash<Value*>()(std::get<0>(t).get()) ^
//       (std::hash<Value*>()(std::get<1>(t).get()) << 1);
//   }
// };

class Value {
public:
  Value() : id(++maxID) {
    // std::cout << "construct a Value" << id << std::endl;
  }
  Value(double a_val, ModelPara a_modelPara = ModelPara::notModelPara)
      : id(++maxID), val(a_val), modelPara(a_modelPara) {
    // std::cout << "construct a Value++" << id << std::endl;
  }

  // 拷贝构造会增加maxID，即是一个新的Value对象
  Value(const Value& other)
      : id(++maxID),
        val(other.val),
        derivative(other.derivative),
        op(other.op),
        prev_(other.prev_),
        modelPara(other.modelPara) {
    // info("copy construct called");
  }

  Value(Value&& other)
      : id(other.id),
        val(other.val),
        derivative(other.derivative),
        op(other.op),
        prev_(std::move(other.prev_)),
        modelPara(other.modelPara) {
    other.id = 0;
    other.val = 0;
    other.derivative = 0;
    other.op = Operation::INVALID;
    other.prev_.clear();
    other.modelPara = ModelPara::notModelPara;

    info("move construct called");
  }

  Value& operator=(Value&& other) {
    if (this != &other) {
      id = other.id;
      val = other.val;
      derivative = other.derivative;
      op = other.op;
      prev_ = std::move(other.prev_);
      modelPara = other.modelPara;

      other.id = 0;
      other.val = 0;
      other.derivative = 0;
      other.op = Operation::INVALID;
      other.prev_.clear();
      other.modelPara = ModelPara::notModelPara;

      info("move= construct called");
    }
    return *this;
  }

  ~Value() {
    // --maxID;   // 先注释掉，放置计算过程中析构导致后续再构建Value出现重复的ID。
    // std::cout << "destory a Value: " << id << std::endl;
  }

  ValuePtr clone() { return std::make_shared<Value>(*this); }

  void backward();

  // 声明成友元的好处：
  // 两个参数，可以实现double + Value，否则类的成员函数第一个参数是this，只能实现Value+double
  // 所有的这些计算操作，都是创建新的res，而不是修改res值
  friend ValuePtr operator+(ValuePtr lhs, ValuePtr rhs);
  friend ValuePtr operator*(ValuePtr lhs, ValuePtr rhs);
  friend ValuePtr operator+(ValuePtr lhs, double num);
  friend ValuePtr operator*(ValuePtr lhs, double num);
  friend ValuePtr operator+(ValuePtr lhs, InputVal iv);
  friend ValuePtr operator*(ValuePtr lhs, InputVal iv);

  friend ValuePtr operator-(ValuePtr lhs, ValuePtr rhs);
  friend ValuePtr operator/(ValuePtr lhs, ValuePtr rhs);

  friend ValuePtr operator-(ValuePtr vp);
  friend ValuePtr inv(ValuePtr vp);
  friend ValuePtr exp(ValuePtr vp);
  friend ValuePtr log(ValuePtr vp);
  friend ValuePtr pow(ValuePtr lhs, double num);
  // friend ValuePtr pow(ValuePtr lhs, InputVal iv);
  friend ValuePtr pow(ValuePtr lhs, ValuePtr rhs);
  friend ValuePtr tanh(ValuePtr vp);
  friend ValuePtr relu(ValuePtr vp);

  friend ValuePtr sum(std::vector<ValuePtr> vvp);
  friend ValuePtr avg(std::vector<ValuePtr> vvp);

  friend ValuePtr operator*(std::vector<ValuePtr> lhs, std::vector<ValuePtr> rhs);
  friend ValuePtr operator*(std::vector<ValuePtr> lhs, std::vector<InputVal> rhs);
  friend ValuePtr wMulXAddB(std::vector<ValuePtr> W, std::vector<ValuePtr> X, ValuePtr b);
  friend ValuePtr wMulXAddB(const std::vector<ValuePtr>& W, const std::vector<InputVal>& X,
                            const ValuePtr& b);
  // friend std::vector<ValuePtr> wMulXAddBBatch(const std::vector<ValuePtr>& W,
  //                                             const std::vector<std::vector<InputVal>>& X,
  //                                             const ValuePtr& b);
  friend ValuePtr regLoss(const std::vector<ValuePtr>& parameters);
  friend ValuePtr hingeLoss(const std::vector<std::vector<ValuePtr>>& yOut,
                            const std::vector<ValuePtr>& yT);
  friend ValuePtr logitLoss(const std::vector<ValuePtr>& yOut, int yT);

  std::string toString() {
    std::stringstream ss;
    ss << "data=" << this->val << " | grad=" << this->derivative;
    return ss.str();
  }

public:
  static int maxID;  // 当创建的Value很多的时候，这个ID可能超过最大值
  static thread_local std::unordered_map<std::pair<ValuePtr, ValuePtr>, ValuePtr, PairHash> cache;
  static ValuePtr placeHolder;
  static ValuePtr placeHolder2;
  static std::mutex mtx;
  static TrainOrEval s_trainOrEval;

public:
  int id = 0;
  double val = 0;
  double derivative = 0;
  ModelPara modelPara = ModelPara::notModelPara;
  Operation op = Operation::INVALID;
  std::vector<ValuePtr> prev_;
};

class InputVal {
public:
  InputVal() : val(0.0) {}
  InputVal(double val_) : val(val_) {}

public:
  double val;
};

std::ostream& operator<<(std::ostream& os, const std::vector<std::shared_ptr<Value>>& vec);

bool operator==(const ValuePtr& lhs, const ValuePtr& rhs);
bool operator<(const ValuePtr& lhs, const ValuePtr& rhs);

struct ValuePtrHash {
  size_t operator()(const ValuePtr& vp) const {
    return std::hash<int>()(vp->id);  // 返回哈希值
  }
};

// 还没有用到，目前还是使用运算符重载
struct ValuePtrEqual {
  bool operator()(const ValuePtr& a, const ValuePtr& b) const { return a->id == b->id; }
};

struct ValuePtrLess {
  bool operator()(const ValuePtr& a, const ValuePtr& b) const { return a->id < b->id; }
};
