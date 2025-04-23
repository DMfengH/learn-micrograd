#include "engine.h"

int Value::maxID = 0;
thread_local std::unordered_map<std::pair<ValuePtr, ValuePtr>, ValuePtr, PairHash> Value::cache;

// 用于单操作数运算符的cache查询；
// placeHolder2是因为同一个Value会参与多个运算符，例如：模型的W还要进行正则化损失的计算，即pow。
ValuePtr Value::placeHolder = std::make_shared<Value>();
ValuePtr Value::placeHolder2 = std::make_shared<Value>();

TrainOrEval Value::s_trainOrEval = TrainOrEval::train;
std::mutex Value::mtx;

std::string toString(Operation op) {
  static const std::unordered_map<Operation, std::string> opMap = {
      {Operation::ADD, "ADD"},
      {Operation::SUBTRACT, "SUBTRACT"},
      {Operation::MULTIPLY, "MULTIPLY"},
      {Operation::DIVIDE, "DIVIDE"},
      {Operation::TANH, "TANH"},
      {Operation::NEG, "NEG"},
      {Operation::INV, "INV"},
      {Operation::EXP, "EXP"},
      {Operation::LOG, "LOG"},
      {Operation::POW, "POW"},
      {Operation::RELU, "RELU"},

      {Operation::SUM, "SUM"},
      {Operation::AVG, "AVG"},

      {Operation::MULI, "MULI"},
      {Operation::MATMUL, "MATMUL"},
      {Operation::MATMULI, "MATMULI"},
      {Operation::HINGELOSS, "HINGELOSS"},
      {Operation::REGLOSS, "REGLOSS"},
      {Operation::LOGITLOSS, "LOGITLOSS"},
      {Operation::WMULXADDB, "WMULXADDB"},
      {Operation::WMULXADDBI, "WMULXADDBI"},

  };

  auto it = opMap.find(op);
  return (it != opMap.end()) ? it->second : "UNKNOWN";
}

template <typename T>
T relu(T x) {
  return std::max(T(0), x);
}

// 这里为什么使用智能指针？
// 因为Node的成员中有指向其他Node的指针 ×
// 新创建的Node对象，需要延长生命周期；使用shared_ptr让out共享所有权就比较好。
// 另外，捕获的内容会变化，所以只能是引用捕获，或者是值捕获指针,而引用捕获会遇到局部变量销毁的问题，所以最好用指针。
ValuePtr operator+(ValuePtr lhs, ValuePtr rhs) {
  ValuePtr out;
  double num = lhs->val + rhs->val;

  if (Value::s_trainOrEval == TrainOrEval::train) {
    out = std::make_shared<Value>(num);
    out->op = Operation::ADD;
    out->prev_.push_back(lhs);
    out->prev_.push_back(rhs);
  } else {
    auto check = std::make_pair(lhs, rhs);
    if (Value::cache.find(check) != Value::cache.end()) {
      out = Value::cache[check];
      out->val = num;
      out->derivative = 0;
    } else {
      out = std::make_shared<Value>(num);
      out->op = Operation::ADD;
      out->prev_.push_back(lhs);
      out->prev_.push_back(rhs);

      Value::cache.insert({check, out});
    }
  }

  return out;
}

ValuePtr operator*(ValuePtr lhs, ValuePtr rhs) {
  ValuePtr out;
  double num = lhs->val * rhs->val;

  if (Value::s_trainOrEval == TrainOrEval::train) {
    out = std::make_shared<Value>(num);
    out->op = Operation::MULTIPLY;
    out->prev_.push_back(lhs);
    out->prev_.push_back(rhs);
  } else {
    auto check = std::make_pair(lhs, rhs);
    if (Value::cache.find(check) != Value::cache.end()) {
      out = Value::cache[check];
      out->val = num;
      out->derivative = 0;
    } else {
      out = std::make_shared<Value>(num);
      out->op = Operation::MULTIPLY;
      out->prev_.push_back(lhs);
      out->prev_.push_back(rhs);

      Value::cache.insert({check, out});
    }
  }

  return out;
}

// 这里这个num的所有权应该在哪里？？？最终是在它计算的out那里
// 为了把num传出去，只能使用智能指针
// 离开作用域还能用，就得是heap上的变量或者拷贝一份，方便管理所以用智能指针
// 只支持和一个num相加，如何一个循环内lhs和多个num相加，会导致多个num提取到的都是同一个check
ValuePtr operator+(ValuePtr lhs, double num) {
  ValuePtr rhsNum;

  if (Value::s_trainOrEval == TrainOrEval::train) {
    rhsNum = std::make_shared<Value>(num);
  } else {
    auto check = std::make_pair(lhs, Value::placeHolder);
    if (Value::cache.find(check) != Value::cache.end()) {
      rhsNum = Value::cache[check];
      rhsNum->val = num;
    } else {
      rhsNum = std::make_shared<Value>(num);

      Value::cache.insert({check, rhsNum});
    }
  }
  return lhs + rhsNum;
}

ValuePtr operator*(ValuePtr lhs, double num) {
  ValuePtr rhsNum;
  if (Value::s_trainOrEval == TrainOrEval::train) {
    rhsNum = std::make_shared<Value>(num);
  } else {
    auto check = std::make_pair(lhs, Value::placeHolder);
    if (Value::cache.find(check) != Value::cache.end()) {
      rhsNum = Value::cache[check];
      rhsNum->val = num;
    } else {
      rhsNum = std::make_shared<Value>(num);

      Value::cache.insert({check, rhsNum});
    }
  }
  return lhs * rhsNum;
}

ValuePtr operator*(ValuePtr lhs, InputVal iv) {
  ValuePtr out;
  double num = lhs->val * iv.val;

  if (Value::s_trainOrEval == TrainOrEval::train) {
    out = std::make_shared<Value>(num);
    out->op = Operation::MULI;
    out->prev_.push_back(lhs);
    out->prev_.push_back(std::make_shared<Value>(iv.val));
  } else {
    auto check = std::make_pair(lhs, Value::placeHolder);
    if (Value::cache.find(check) != Value::cache.end()) {
      out = Value::cache[check];
      out->val = num;
      out->derivative = 0;
      out->prev_[1]->val = iv.val;
    } else {
      out = std::make_shared<Value>(num);
      out->op = Operation::MULI;
      out->prev_.push_back(lhs);
      out->prev_.push_back(std::make_shared<Value>(iv.val));

      Value::cache.insert({check, out});
    }
  }
  return out;
}

// tanh的计算有多种方式
// 每个value只能进行一个这种单操作数的操作
ValuePtr tanh(ValuePtr vp) {
  ValuePtr out;
  double num = (pow(M_E, 2 * vp->val) - 1) / (pow(M_E, 2 * vp->val) + 1);

  if (Value::s_trainOrEval == TrainOrEval::train) {
    out = std::make_shared<Value>(num);
    out->op = Operation::TANH;
    out->derivative = 0;
    out->prev_.push_back(vp);
  } else {
    auto check = std::make_pair(vp, Value::placeHolder);
    if (Value::cache.find(check) != Value::cache.end()) {
      out = Value::cache[check];
      out->val = num;
      out->derivative = 0;
    } else {
      out = std::make_shared<Value>(num);
      out->op = Operation::TANH;
      out->derivative = 0;
      out->prev_.push_back(vp);

      Value::cache.insert({check, out});
    }
  }

  return out;
}

ValuePtr operator-(ValuePtr vp) {
  ValuePtr out;
  double num = -vp->val;

  if (Value::s_trainOrEval == TrainOrEval::train) {
    out = std::make_shared<Value>(num);
    out->op = Operation::NEG;
    out->prev_.push_back(vp);
  } else {
    auto check = std::make_pair(vp, Value::placeHolder);
    if (Value::cache.find(check) != Value::cache.end()) {
      out = Value::cache[check];
      out->val = num;
      out->derivative = 0;
    } else {
      out = std::make_shared<Value>(num);
      out->op = Operation::NEG;
      out->prev_.push_back(vp);

      Value::cache.insert({check, out});
    }
  }
  return out;
}

ValuePtr operator-(ValuePtr lhs, ValuePtr rhs) {
  ValuePtr inter = -rhs;
  return lhs + inter;
}

ValuePtr inv(ValuePtr vp) {
  ValuePtr out;
  double num = pow(vp->val, -1);

  if (Value::s_trainOrEval == TrainOrEval::train) {
    out = std::make_shared<Value>(num);
    out->op = Operation::INV;
    out->prev_.push_back(vp);
  } else {
    auto check = std::make_pair(vp, Value::placeHolder);
    if (Value::cache.find(check) != Value::cache.end()) {
      out = Value::cache[check];
      out->val = num;
      out->derivative = 0;
    } else {
      out = std::make_shared<Value>(num);
      out->op = Operation::INV;
      out->prev_.push_back(vp);

      Value::cache.insert({check, out});
    }
  }
  return out;
}

ValuePtr operator/(ValuePtr lhs, ValuePtr rhs) {
  ValuePtr inter = inv(rhs);
  // ValuePtr inter = pow(rhs,-1);

  return lhs * inter;
}

ValuePtr exp(ValuePtr vp) {
  ValuePtr out;
  double num = exp(vp->val);
  if (Value::s_trainOrEval == TrainOrEval::train) {
    out = std::make_shared<Value>(num);
    out->op = Operation::EXP;
    out->prev_.push_back(vp);
  } else {
    auto check = std::make_pair(vp, Value::placeHolder);
    if (Value::cache.find(check) != Value::cache.end()) {
      out = Value::cache[check];
      out->val = num;
      out->derivative = 0;
    } else {
      out = std::make_shared<Value>(num);
      out->op = Operation::EXP;
      out->prev_.push_back(vp);

      Value::cache.insert({check, out});
    }
  }
  return out;
}

ValuePtr log(ValuePtr vp) {
  ValuePtr out;
  double num = log(vp->val);

  if (Value::s_trainOrEval == TrainOrEval::train) {
    out = std::make_shared<Value>(num);
    out->op = Operation::LOG;
    out->prev_.push_back(vp);
  } else {
    auto check = std::make_pair(vp, Value::placeHolder);
    if (Value::cache.find(check) != Value::cache.end()) {
      out = Value::cache[check];
      out->val = num;
      out->derivative = 0;
    } else {
      out = std::make_shared<Value>(num);
      out->op = Operation::LOG;
      out->prev_.push_back(vp);

      Value::cache.insert({check, out});
    }
  }
  return out;
}

ValuePtr pow(ValuePtr lhs, ValuePtr rhs) {
  ValuePtr out;
  double num = pow(lhs->val, rhs->val);

  if (Value::s_trainOrEval == TrainOrEval::train) {
    out = std::make_shared<Value>(num);
    out->op = Operation::POW;
    out->prev_.push_back(lhs);
    out->prev_.push_back(rhs);
  } else {
    auto check = std::make_pair(lhs, rhs);
    if (Value::cache.find(check) != Value::cache.end()) {
      out = Value::cache[check];
      out->val = num;
      out->derivative = 0;
    } else {
      out = std::make_shared<Value>(num);
      out->op = Operation::POW;
      out->prev_.push_back(lhs);
      out->prev_.push_back(rhs);

      Value::cache.insert({check, out});
    }
  }
  return out;
}

ValuePtr pow(ValuePtr lhs, double rhs) {
  ValuePtr rhsNum;
  double num = rhs;

  if (Value::s_trainOrEval == TrainOrEval::train) {
    rhsNum = std::make_shared<Value>(num);
  } else {
    auto check = std::make_pair(lhs, Value::placeHolder2);
    if (Value::cache.find(check) != Value::cache.end()) {
      rhsNum = Value::cache[check];
      rhsNum->val = num;
    } else {
      rhsNum = std::make_shared<Value>(num);
      Value::cache.insert({check, rhsNum});
    }
  }
  return pow(lhs, rhsNum);
}

// 暂时先改成leakyrelu
ValuePtr relu(ValuePtr vp) {
  ValuePtr out;
  double num;
  if (vp->val < 0) {
    // num = 0.01 * vp->val;
    num = 0.0;
  } else {
    num = vp->val;
  }

  if (Value::s_trainOrEval == TrainOrEval::train) {
    out = std::make_shared<Value>(num);
    out->op = Operation::RELU;
    out->prev_.push_back(vp);
  } else {
    auto check = std::make_pair(vp, Value::placeHolder);
    if (Value::cache.find(check) != Value::cache.end()) {
      out = Value::cache[check];
      out->val = num;
      out->derivative = 0;
    } else {
      out = std::make_shared<Value>(num);
      out->op = Operation::RELU;
      out->prev_.push_back(vp);

      Value::cache.insert({check, out});
    }
  }
  return out;
}

ValuePtr sum(std::vector<ValuePtr> vvp) {
  ValuePtr out;
  double num = 0.0;
  for (size_t i = 0; i < vvp.size(); ++i) {
    num += vvp[i]->val;
  }

  if (Value::s_trainOrEval == TrainOrEval::train) {
    out = std::make_shared<Value>(num);
    out->op = Operation::SUM;
    for (size_t i = 0; i < vvp.size(); ++i) {
      out->prev_.push_back(vvp[i]);
    }
  }

  return out;
}
ValuePtr avg(std::vector<ValuePtr> vvp) {
  ValuePtr out;
  double num = 0.0;
  for (size_t i = 0; i < vvp.size(); ++i) {
    num += vvp[i]->val;
  }
  num = num / vvp.size();

  if (Value::s_trainOrEval == TrainOrEval::train) {
    out = std::make_shared<Value>(num);
    out->op = Operation::AVG;
    for (size_t i = 0; i < vvp.size(); ++i) {
      out->prev_.push_back(vvp[i]);
    }
  }
  return out;
}

ValuePtr operator*(std::vector<ValuePtr> lhs, std::vector<ValuePtr> rhs) {
  ValuePtr out;
  if (lhs.size() != rhs.size()) {
    error("two operators' degress are not same");
  };

  double num = 0;
  for (size_t i = 0; i < lhs.size(); ++i) {
    num += (lhs[i]->val * rhs[i]->val);
  }

  if (Value::s_trainOrEval == TrainOrEval::train) {
    out = std::make_shared<Value>(num);
    out->op = Operation::MATMUL;
    for (size_t i = 0; i < lhs.size(); ++i) {
      out->prev_.push_back(lhs[i]);
      out->prev_.push_back(rhs[i]);
    }
  }

  return out;
}

ValuePtr operator*(std::vector<ValuePtr> lhs, std::vector<InputVal> rhs) {
  ValuePtr out;
  if (lhs.size() != rhs.size()) {
    error("two operators' degress are not same");
  };

  double num = 0;
  for (size_t i = 0; i < lhs.size(); ++i) {
    num += (lhs[i]->val * rhs[i].val);
  }

  if (Value::s_trainOrEval == TrainOrEval::train) {
    out = std::make_shared<Value>(num);
    out->op = Operation::MATMULI;
    for (size_t i = 0; i < lhs.size(); ++i) {
      out->prev_.push_back(lhs[i]);
      out->prev_.push_back(std::make_shared<Value>(rhs[i].val));
    }
  }

  return out;
}

ValuePtr wMulXAddB(std::vector<ValuePtr> W, std::vector<ValuePtr> X, ValuePtr b) {
  ValuePtr out;
  double num = b->val;
  for (size_t i = 0; i < W.size(); ++i) {
    num += W[i]->val * X[i]->val;
  }

  if (Value::s_trainOrEval == TrainOrEval::train) {
    out = std::make_shared<Value>(num);
    out->op = Operation::WMULXADDB;
    for (size_t i = 0; i < W.size(); ++i) {
      out->prev_.push_back(W[i]);
      out->prev_.push_back(X[i]);
    }
    out->prev_.push_back(b);
  }

  return out;
}

ValuePtr wMulXAddB(const std::vector<ValuePtr>& W, const std::vector<InputVal>& X,
                   const ValuePtr& b) {
  ValuePtr out;
  // double num = std::transform_reduce(W.begin(), W.end(), X.begin(), b->val, std::plus<>(),
  //                                    [](const auto& w, const auto& x) { return w->val * x.val;
  //                                    });
  double num = b->val;
  for (size_t i = 0; i < W.size(); ++i) {
    num += W[i]->val * X[i].val;
  }

  if (Value::s_trainOrEval == TrainOrEval::train) {
    out = std::make_shared<Value>(num);
    out->op = Operation::WMULXADDBI;
    for (size_t i = 0; i < W.size(); ++i) {
      out->prev_.push_back(W[i]);
      out->prev_.push_back(std::make_shared<Value>(X[i].val, ModelPara::input));
    }
    out->prev_.push_back(b);
  }

  return out;
}

// std::vector<ValuePtr> wMulXAddBBatch(const std::vector<ValuePtr>& W,
//                                      const std::vector<std::vector<InputVal>>& X,
//                                      const ValuePtr& b) {
//   std::vector<ValuePtr> out;
// }

ValuePtr hingeLoss(const std::vector<std::vector<ValuePtr>>& yOut,
                   const std::vector<ValuePtr>& yT) {
  ValuePtr out;
  double num = 0;
  for (size_t i = 0; i < yOut.size(); ++i) {
    num += relu((-(yT[i]->val * yOut[i][0]->val) + 1));
  }
  num = num / yOut.size();
  if (Value::s_trainOrEval == TrainOrEval::train) {
    out = std::make_shared<Value>(num);
    out->op = Operation::HINGELOSS;
    for (size_t i = 0; i < yOut.size(); ++i) {
      out->prev_.push_back(yOut[i][0]);
      out->prev_.push_back(yT[i]);
    }
  }

  return out;
}

ValuePtr regLoss(const std::vector<ValuePtr>& parameters) {
  ValuePtr out;
  double num = 0.0;
  for (size_t i = 0; i < parameters.size(); ++i) {
    num += pow(parameters[i]->val, 2);
  }
  if (Value::s_trainOrEval == TrainOrEval::train) {
    out = std::make_shared<Value>(num);
    out->op = Operation::REGLOSS;
    for (size_t i = 0; i < parameters.size(); ++i) {
      out->prev_.push_back(parameters[i]);
    }
  }

  return out;
}

// 这个loss写成单个输入的，而不是batch，因为backward有点复杂
ValuePtr logitLoss(const std::vector<ValuePtr>& yOut, int yT) {
  ValuePtr out;
  double num = 0.0;
  double outSum = 0.0;
  for (auto& output : yOut) {
    outSum += exp(output->val);
  }

  num = -log(exp(yOut[yT]->val) * (1 / outSum));

  if (Value::s_trainOrEval == TrainOrEval::train) {
    out = std::make_shared<Value>(num);
    out->op = Operation::LOGITLOSS;
    for (int i = 0; i < yOut.size(); ++i) {
      if (i != yT) {
        out->prev_.push_back(yOut[i]);
      }
    }
    out->prev_.push_back(yOut[yT]);
    // out->prev_.push_back(std::make_shared<Value>(outSum));
  }

  return out;
}

void Value::backward() {
  switch (this->op) {
    case Operation::ADD:
      this->prev_[0]->derivative += this->derivative;
      this->prev_[1]->derivative += this->derivative;
      break;
    case Operation::SUBTRACT:
      break;
    case Operation::MULTIPLY:
      this->prev_[0]->derivative += this->prev_[1]->val * this->derivative;
      this->prev_[1]->derivative += this->prev_[0]->val * this->derivative;
      break;
    case Operation::DIVIDE:
      break;
    case Operation::TANH:
      this->prev_[0]->derivative += (1 - pow(this->val, 2)) * this->derivative;
      break;
    case Operation::NEG:
      this->prev_[0]->derivative += -1 * this->derivative;
      break;
    case Operation::INV:
      this->prev_[0]->derivative += -1 * pow(this->prev_[0]->val, -2) * this->derivative;
      break;
    case Operation::EXP:
      this->prev_[0]->derivative += this->val * this->derivative;
      break;
    case Operation::LOG:
      this->prev_[0]->derivative += (1 / this->prev_[0]->val) * this->derivative;
      break;
    case Operation::POW:
      this->prev_[0]->derivative += (this->prev_[1]->val) *
                                    pow(this->prev_[0]->val, this->prev_[1]->val - 1) *
                                    this->derivative;
      this->prev_[1]->derivative += this->val * log(this->prev_[0]->val) * this->derivative;
      break;
    case Operation::RELU:
      if (this->val > 0) {
        this->prev_[0]->derivative += this->derivative;
      } else {
        // this->prev_[0]->derivative += 0.01 * this->derivative;
        this->prev_[0]->derivative += 0.0;
      }
      break;
    case Operation::ADDI:
      this->prev_[0]->derivative += this->derivative;
      break;
    case Operation::SUM:
      for (size_t i = 0; i < this->prev_.size(); ++i) {
        this->prev_[i]->derivative += this->derivative;
      }
      break;
    case Operation::AVG: {
      size_t num = this->prev_.size();
      for (size_t i = 0; i < num; ++i) {
        this->prev_[i]->derivative += (1.0 / num) * this->derivative;
      }
      break;
    }
    case Operation::MULI:
      this->prev_[0]->derivative += this->prev_[1]->val * this->derivative;
      break;
    case Operation::MATMUL:
      for (size_t i = 0; i < this->prev_.size() / 2; ++i) {
        this->prev_[2 * i]->derivative += this->prev_[2 * i + 1]->val * this->derivative;
        this->prev_[2 * i + 1]->derivative += this->prev_[2 * i]->val * this->derivative;
      }
      break;
    case Operation::MATMULI:
      for (size_t i = 0; i < this->prev_.size() / 2; ++i) {
        this->prev_[2 * i]->derivative += this->prev_[2 * i + 1]->val * this->derivative;
      }
      break;
    case Operation::HINGELOSS:
      for (size_t i = 0; i < this->prev_.size() / 2; ++i) {
        if (this->prev_[2 * i]->val * this->prev_[2 * i + 1]->val > 1) {
          this->prev_[2 * i]->derivative += 0.0;
          this->prev_[2 * i + 1]->derivative += 0.0;
        } else {
          double size = this->prev_.size() / 2;
          this->prev_[2 * i]->derivative +=
              (-1 / size) * this->prev_[2 * i + 1]->val * this->derivative;
          this->prev_[2 * i + 1]->derivative +=
              (-1 / size) * this->prev_[2 * i]->val * this->derivative;
        }
      }
      break;
    case Operation::REGLOSS:
      for (size_t i = 0; i < this->prev_.size(); ++i) {
        this->prev_[i]->derivative += 2 * this->prev_[i]->val * this->derivative;
      }
      break;
    case Operation::LOGITLOSS: {
      // this->prev_[0]->derivative += ((exp(this->prev_[0]->val) / this->prev_[1]->val) - 1.0) *
      // this->derivative;
      size_t num = this->prev_.size();
      double outSum = 0.0;
      for (size_t i = 0; i < num; ++i) {
        outSum += exp(this->prev_[i]->val);
      }
      for (size_t i = 0; i < num - 1; ++i) {
        this->prev_[i]->derivative += (exp(this->prev_[i]->val) / outSum) * this->derivative;
      }
      this->prev_[num - 1]->derivative +=
          ((exp(this->prev_[num - 1]->val) / outSum) - 1.0) * this->derivative;

      break;
    }
    case Operation::WMULXADDB:
      for (size_t i = 0; i < this->prev_.size() / 2; ++i) {
        this->prev_[2 * i]->derivative += this->prev_[2 * i + 1]->val * this->derivative;
        this->prev_[2 * i + 1]->derivative += this->prev_[2 * i]->val * this->derivative;
      }
      this->prev_.back()->derivative += this->derivative;
      break;
    case Operation::WMULXADDBI:
      for (size_t i = 0; i < this->prev_.size() / 2; ++i) {
        this->prev_[2 * i]->derivative += this->prev_[2 * i + 1]->val * this->derivative;
      }
      this->prev_.back()->derivative += this->derivative;
      break;
    default:
      warn("无效输入---------");
  }
}

std::ostream& operator<<(std::ostream& os, const std::vector<std::shared_ptr<Value>>& vec) {
  os << "\n[";
  for (size_t i = 0; i < vec.size(); ++i) {
    os << vec[i]->val;
    if (i < vec.size() - 1) {
      os << ", ";
    }
  }
  os << "]";

  return os;
}

bool operator==(const ValuePtr& lhs, const ValuePtr& rhs) {
  // info(" operator == is invoked");
  return lhs->id == rhs->id;
}
bool operator<(const ValuePtr& lhs, const ValuePtr& rhs) {
  // info(" operator < is invoked");
  return lhs->id < rhs->id;
};
