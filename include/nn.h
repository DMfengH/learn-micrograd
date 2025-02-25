#pragma once
#include "engine.h"
#include "utils.h"

#include <random>

using Logger::info;
using Logger::warn;

// 这里有个需要关注的点：每次input经过Neuron，layer，MLP，都会创建新的节点，
//   所以要区分，什么时候创建节点并构建节点间的关系，什么时候只是修改节点的值
class Neuron{
public:
  Neuron(){
    // info("construct a Neuron");
  }
  // 默认参数只能写在声明这里，不能写在定义中
  Neuron(int indegree_, bool nonLin_=true);

  void print();

  std::unique_ptr<ValuePtr[]> parameters();

  ValuePtr operator()(const std::unique_ptr<ValuePtr[]>& input);
  
public:
  static std::mt19937 gen;  // 选择 Mersenne Twister 作为随机引擎
  static std::uniform_real_distribution<double> dist;

  size_t indegree = 0;
  std::unique_ptr<ValuePtr[]> W = nullptr; 
  ValuePtr b = nullptr;
  bool nonLin = true;
};

class Layer{
public:
  Layer(){
    // info("construct a layer");
  }
  Layer(int inDegree_, int outDegree_, bool nonLin=true);

  // 这里如果使用vector，应该会简单很多
  std::unique_ptr<ValuePtr[]> parameters();

  std::unique_ptr<ValuePtr[]> operator()(const std::unique_ptr<ValuePtr[]>& input);

public:
  std::unique_ptr<Neuron[]> ns = nullptr;
  size_t inDegree = 0;
  size_t outDegree = 0;

};

class MLP{
public:
  MLP(int inDegree_, int numLayers_, int* outDegrees_);

  std::unique_ptr<ValuePtr[]> parameters();

  std::unique_ptr<ValuePtr[]> operator()(const std::unique_ptr<ValuePtr[]>& input);

public:
  size_t inDegree=0;
  size_t numLayers=0;
  int* outDegrees = nullptr;
  std::unique_ptr<Layer[]> layers = nullptr;

};
