#include "nn.h"

std::mt19937 Neuron::gen(std::random_device{}());
std::uniform_real_distribution<double> Neuron::dist(-1.0, 1.0);

Neuron::Neuron(int indegree_, bool nonLin_): indegree(indegree_), 
                                                  W(std::make_unique<ValuePtr[]>(indegree_)),
                                                  nonLin(nonLin_){
    for(int i=0; i<indegree; i++){
      W[i] = std::make_shared<Value>(dist(gen));     // W 访问成员不需要->，感觉怪怪的？实际上，里面重载了[]操作符
    }
    b = std::make_shared<Value>(0);
}

void Neuron::print(){
  std::stringstream ss;
  ss << "Neuron{W: ";
  for(int i=0; i<this->indegree; i++){
    ss << W[i]->val << " ";
  }
  ss << "b: " << b->val << "}\n";
  info(ss.str());
}

std::unique_ptr<ValuePtr[]> Neuron::parameters(){
  std::unique_ptr<ValuePtr[]> paras = std::make_unique<ValuePtr[]>(indegree+1);
  for(size_t i=0; i<indegree; i++){
    paras[i] = W[i];
  }
  paras[indegree] = b;
  
  return paras;
}

// unique_ptr是不能拷贝的，所以不能直接作为参数
ValuePtr Neuron::operator()(const std::unique_ptr<ValuePtr[]>& input){
  ValuePtr res = b;       // 这个还没有存储到cache中
  for (int i=0; i<indegree; i++){
    res = res + W[i]*input[i]; 
  }

  if(nonLin){
    // res = tanh(res);
    res = relu(res);
  }

  return res;
}

Layer::Layer(int inDegree_, int outDegree_, bool nonLin): ns(std::make_unique<Neuron[]>(outDegree_)), 
                                                                inDegree(inDegree_), 
                                                                outDegree(outDegree_){
  double XavierInitRange = std::sqrt(6) / std::sqrt(inDegree+outDegree);
  Neuron::dist.param(std::uniform_real_distribution<double>::param_type(-XavierInitRange, XavierInitRange));
  
  for (int i=0; i<outDegree; i++){
    ns[i] = Neuron(inDegree,nonLin);
  }
}

std::unique_ptr<ValuePtr[]> Layer::parameters(){
  size_t numParas= (inDegree+1)* outDegree;
  std::unique_ptr<ValuePtr[]> paras = std::make_unique<ValuePtr[]>(numParas);
  for (size_t i=0; i < numParas; i++){
    paras[i] = ns[i/(inDegree+1)].parameters()[i%(inDegree+1)];
  }

  return paras;
}

std::unique_ptr<ValuePtr[]> Layer::operator()(const std::unique_ptr<ValuePtr[]>& input){
  std::unique_ptr<ValuePtr[]> out = std::make_unique<ValuePtr[]>(outDegree);
  for(int i=0; i< outDegree; i++){
    out[i] = ns[i](input);
  } 

  return out;
}

MLP::MLP(int inDegree_, int numLayers_, int* outDegrees_): inDegree(inDegree_), 
                                                        numLayers(numLayers_), 
                                                        outDegrees(outDegrees_),    
                                                        layers(std::make_unique<Layer[]>(numLayers_)){
  layers[0] = Layer(inDegree, outDegrees[0]);          
  for (size_t i =1; i <numLayers;i++){
    if (i == numLayers-1){
      layers[i] = Layer(outDegrees[i-1], outDegrees[i], false);
    }else{
      layers[i] = Layer(outDegrees[i-1], outDegrees[i], true);
    }
  }
}

std::unique_ptr<ValuePtr[]> MLP::parameters(){
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

std::unique_ptr<ValuePtr[]> MLP::operator()(const std::unique_ptr<ValuePtr[]>& input){
  std::unique_ptr<ValuePtr[]> res;
  res = layers[0](input);
  for (size_t i=1; i<numLayers; i++){
    res = layers[i](res);
  }
  
  return res;
}