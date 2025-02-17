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
  Neuron(){info("construct a Neuron");}
  Neuron(int indegree_): indegree(indegree_), W(std::make_unique<ValuePtr[]>(indegree_)){
    for(int i=0; i<indegree; i++){
      W[i] = std::make_shared<Value>(dist(gen));     // W 访问成员不需要->，感觉怪怪的？实际上，里面重载了[]操作符
      
    }
    b = std::make_shared<Value>(dist(gen));
  }

  void print(){
    std::stringstream ss;
    ss << "Neuron{W: ";
    for(int i=0; i<this->indegree; i++){
      ss << W[i]->val << " ";
    }
    ss << "b: " << b->val << "}\n";
    info(ss.str());
  }

  std::unique_ptr<ValuePtr[]> parameters(){
    std::unique_ptr<ValuePtr[]> paras = std::make_unique<ValuePtr[]>(indegree+1);
    for(size_t i=0; i<indegree; i++){
      paras[i] = W[i];
    }
    paras[indegree] = b;
    
    return paras;
  }

  // unique_ptr是不能拷贝的，所以不能直接作为参数
  ValuePtr operator()(const std::unique_ptr<ValuePtr[]>& input){
    ValuePtr res = b;
    for (int i=0; i<indegree; i++){
      res = res + W[i]*input[i]; 
    }
    res = tanh(res);
    return res;
  }
  

public:
  static std::mt19937 gen;  // 选择 Mersenne Twister 作为随机引擎
  static std::uniform_real_distribution<double> dist;

  size_t indegree = 0;
  std::unique_ptr<ValuePtr[]> W = nullptr; 
  ValuePtr b = nullptr;
};

class Layer{
public:
  Layer(){info("construct a layer");}
  Layer(int inDegree_, int outDegree_): ns(std::make_unique<Neuron[]>(outDegree_)), inDegree(inDegree_), outDegree(outDegree_){
    for (int i=0; i<outDegree; i++){
      ns[i] = Neuron(inDegree);
    }
  }

  // 这里如果使用vector，应该会简单很多
  std::unique_ptr<ValuePtr[]> parameters(){
    size_t numParas= (inDegree+1)* outDegree;
    std::unique_ptr<ValuePtr[]> paras = std::make_unique<ValuePtr[]>(numParas);
    for (size_t i=0; i < numParas; i++){
      paras[i] = ns[i/(inDegree+1)].parameters()[i%(inDegree+1)];
    }

    return paras;
  }

  std::unique_ptr<ValuePtr[]> operator()(const std::unique_ptr<ValuePtr[]>& input){
    std::unique_ptr<ValuePtr[]> out = std::make_unique<ValuePtr[]>(outDegree);
    for(int i=0; i< outDegree; i++){
      out[i] = ns[i](input);
    } 

    return out;
  }


public:
  std::unique_ptr<Neuron[]> ns = nullptr;
  size_t inDegree = 0;
  size_t outDegree = 0;

};

class MLP{
public:
  MLP(int inDegree_, int numLayers_, int* outDegrees_): inDegree(inDegree_), 
                                                        numLayers(numLayers_), 
                                                        outDegrees(outDegrees_),    
                                                        layers(std::make_unique<Layer[]>(numLayers_)){
    layers[0] = Layer(inDegree, outDegrees[0]);          
    for (size_t i =1; i <numLayers;i++){
      layers[i] = Layer(outDegrees[i-1], outDegrees[i]);
    }
  }

  std::unique_ptr<ValuePtr[]> parameters(){
    size_t numParas=0;
    numParas += (inDegree+1)*outDegrees[0];
    for (size_t i=0; i< numLayers-1; i++){
      numParas += (outDegrees[i]+1)*outDegrees[i+1];
    }
    
    std::unique_ptr<ValuePtr[]> paras = std::make_unique<ValuePtr[]>(numParas);
    size_t index = 0;
    for (size_t i=0; i<numLayers; i++){
      Layer& layer = layers[i];
      size_t numNeuronPerLayer = (layer.inDegree+1)*layer.outDegree;
      for (size_t j=0; j<numNeuronPerLayer; j++){
        paras[index] = layer.parameters()[j];
        index +=1;
      }
    }
    
    return paras;
  }

  std::unique_ptr<ValuePtr[]> operator()(const std::unique_ptr<ValuePtr[]>& input){
    std::unique_ptr<ValuePtr[]> res;
    res = layers[0](input);
    for (size_t i=1; i<numLayers; i++){
      res = layers[i](res);
    }
    
    return res;
  }

public:
  size_t inDegree=0;
  size_t numLayers=0;
  int* outDegrees = nullptr;
  std::unique_ptr<Layer[]> layers = nullptr;

};
