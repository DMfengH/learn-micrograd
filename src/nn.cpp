#include "nn.h"

std::mt19937 Neuron::gen(std::random_device{}());
std::normal_distribution<double> Neuron::dist(0, 1.0);

Neuron::Neuron(int indegree_, bool nonLin_) : indegree(indegree_), nonLin(nonLin_) {
  W.reserve(indegree);
  for (int i = 0; i < indegree; i++) {
    W.emplace_back(std::make_shared<Value>(dist(gen)));
  }
  b = std::make_shared<Value>(0);
}

void Neuron::print() {
  std::stringstream ss;
  ss << "Neuron{W: ";
  for (int i = 0; i < this->indegree; i++) {
    ss << W[i]->val << " ";
  }
  ss << "b: " << b->val << "}\n";
  info(ss.str());
}

std::vector<ValuePtr> Neuron::parameters() {
  std::vector<ValuePtr> paras;
  paras.reserve(indegree + 1);

  for (size_t i = 0; i < indegree; i++) {
    paras.push_back(W[i]);
  }
  paras.push_back(b);

  return paras;
}

// unique_ptr是不能拷贝的，所以不能直接作为参数
ValuePtr Neuron::operator()(const std::vector<ValuePtr>& input) {
  ValuePtr res = b;
  for (int i = 0; i < indegree; i++) {
    res = res + W[i] * input[i];  // 这种代码如何写两个两个相加，而不是都往res上累加
  }

  if (nonLin) {
    // res = tanh(res);
    res = relu(res);
  }

  return res;
}

ValuePtr Neuron::operator()(const std::vector<InputVal>& input) {
  ValuePtr res = b;
  for (int i = 0; i < indegree; i++) {
    res = res + W[i] * input[i];  // 这种代码如何写两个两个相加，而不是都往res上累加
  }

  if (nonLin) {
    // res = tanh(res);
    res = relu(res);
  }

  return res;
}

Layer::Layer(int inDegree_, int outDegree_, bool nonLin)
    : inDegree(inDegree_), outDegree(outDegree_) {
  double HeInit = std::sqrt(2) / std::sqrt(inDegree);
  info("W Distribute range: ", 0, HeInit);
  Neuron::dist.param(std::normal_distribution<double>::param_type(0, HeInit));
  ns.reserve(outDegree);
  for (int i = 0; i < outDegree; i++) {
    ns.emplace_back(inDegree, nonLin);
  }
}

std::vector<ValuePtr> Layer::parameters() {
  size_t numParas = (inDegree + 1) * outDegree;
  std::vector<ValuePtr> paras;
  paras.reserve(numParas);

  for (size_t i = 0; i < outDegree; i++) {
    // Neuron& n = ns[i];
    const std::vector<ValuePtr>& nv = ns[i].parameters();
    paras.insert(paras.end(), nv.begin(), nv.end());
  }

  return paras;
}

std::vector<ValuePtr> Layer::operator()(const std::vector<ValuePtr>& input) {
  std::vector<ValuePtr> out;
  out.reserve(outDegree);
  for (int i = 0; i < outDegree; i++) {
    out.push_back(ns[i](input));
  }

  return out;
}

std::vector<ValuePtr> Layer::operator()(const std::vector<InputVal>& input) {
  std::vector<ValuePtr> out;
  out.reserve(outDegree);
  for (int i = 0; i < outDegree; i++) {
    out.push_back(ns[i](input));
  }

  return out;
}

MLP::MLP(int inDegree_, int numLayers_, int* outDegrees_)
    : inDegree(inDegree_), numLayers(numLayers_), outDegrees(outDegrees_) {
  layers.reserve(numLayers);
  layers.emplace_back(inDegree, outDegrees[0]);
  for (size_t i = 1; i < numLayers; i++) {
    if (i == numLayers - 1) {
      layers.emplace_back(outDegrees[i - 1], outDegrees[i], false);
    } else {
      layers.emplace_back(outDegrees[i - 1], outDegrees[i], true);
    }
  }
}

std::vector<ValuePtr> MLP::parameters() {
  size_t numParas = 0;
  numParas += (inDegree + 1) * outDegrees[0];
  for (size_t i = 0; i < numLayers - 1; i++) {
    numParas += (outDegrees[i] + 1) * outDegrees[i + 1];
  }

  std::vector<ValuePtr> paras;
  paras.reserve(numParas);

  for (size_t i = 0; i < numLayers; i++) {
    const std::vector<ValuePtr> lv = layers[i].parameters();
    paras.insert(paras.end(), lv.begin(), lv.end());
  }

  return paras;
}

std::vector<ValuePtr> MLP::operator()(const std::vector<InputVal>& input) {
  std::vector<ValuePtr> res;
  res = layers[0](input);
  for (size_t i = 1; i < numLayers; i++) {
    res = layers[i](res);
  }

  return res;
}