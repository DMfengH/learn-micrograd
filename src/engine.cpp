#include "engine.h"

int Value::maxID = 0;
thread_local std::unordered_map<std::pair<ValuePtr, ValuePtr>, ValuePtr, PairHash> Value::cache;
ValuePtr Value::placeHolder = std::make_shared<Value>();
ValuePtr Value::placeHolder2 = std::make_shared<Value>();
std::mutex Value::mtx;

std::string toString(Operation op) {
  static const std::unordered_map<Operation, std::string> opMap = {
      {Operation::ADD, "ADD"},           {Operation::SUBTRACT, "SUBTRACT"},
      {Operation::MULTIPLY, "MULTIPLY"}, {Operation::DIVIDE, "DIVIDE"},
      {Operation::TANH, "TANH"},         {Operation::NEG, "NEG"},
      {Operation::INV, "INV"},           {Operation::EXP, "EXP"},
      {Operation::POW, "POW"},           {Operation::RELU, "RELU"},
      {Operation::MULI, "MULI"},
  };
  auto it = opMap.find(op);
  return (it != opMap.end()) ? it->second : "UNKNOWN";
}

// 这里为什么使用智能指针？
// 因为Node的成员中有指向其他Node的指针 ×
// 新创建的Node对象，需要延长生命周期；使用shared_ptr让out共享所有权就比较好。
// 另外，捕获的内容会变化，所以只能是引用捕获，或者是值捕获指针,而引用捕获会遇到局部变量销毁的问题，所以最好用指针。
ValuePtr operator+(ValuePtr lhs, ValuePtr rhs) {
  // std::lock_guard<std::mutex> lock(Value::mtx);
  ValuePtr out;
  auto check = std::make_pair(lhs, rhs);

  if (Value::cache.find(check) != Value::cache.end()) {
    // info("find a already exist Value");
    out = Value::cache[check];
    out->val = lhs->val + rhs->val;
    out->derivative = 0;
  } else {
    out = std::make_shared<Value>(lhs->val + rhs->val);
    out->op = Operation::ADD;
    out->prev_.push_back(lhs);
    out->prev_.push_back(rhs);

    Value::cache.insert({check, out});
  }

  return out;
}

ValuePtr operator*(ValuePtr lhs, ValuePtr rhs) {
  // std::lock_guard<std::mutex> lock(Value::mtx);

  ValuePtr out;
  auto check = std::make_pair(lhs, rhs);

  if (Value::cache.find(check) != Value::cache.end()) {
    // info("find a already exist Value");
    out = Value::cache[check];
    out->val = lhs->val * rhs->val;
    out->derivative = 0;
  } else {
    double a = lhs->val;
    double b = rhs->val;
    out = std::make_shared<Value>(a * b);
    out->op = Operation::MULTIPLY;
    out->prev_.push_back(lhs);
    out->prev_.push_back(rhs);

    Value::cache.insert({check, out});
  }

  return out;
}

// 这里这个num的所有权应该在哪里？？？最终是在它计算的out那里
// 为了把num传出去，只能使用智能指针
// 离开作用域还能用，就得是heap上的变量或者拷贝一份，方便管理所以用智能指针
// 只支持和一个num相加，如何一个循环内lhs和多个num相加，会导致多个num提取到的都是同一个check
ValuePtr operator+(ValuePtr lhs, double num) {
  ValuePtr rhsNum;
  {
    // std::lock_guard<std::mutex> lock(Value::mtx);

    auto check = std::make_pair(lhs, Value::placeHolder);

    if (Value::cache.find(check) != Value::cache.end()) {
      // info("find a already exist Value");
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
  {
    // std::lock_guard<std::mutex> lock(Value::mtx);

    auto check = std::make_pair(lhs, Value::placeHolder);

    if (Value::cache.find(check) != Value::cache.end()) {
      // info("find a already exist Value");
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
  double num = lhs->val * iv.val;
  auto check = std::make_pair(lhs, Value::placeHolder);
  ValuePtr out;
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
  return out;
}

// tanh的计算有多种方式
// 每个value只能进行一个这种单操作数的操作
ValuePtr tanh(ValuePtr vp) {
  // std::lock_guard<std::mutex> lock(Value::mtx);

  double num = (pow(M_E, 2 * vp->val) - 1) / (pow(M_E, 2 * vp->val) + 1);

  ValuePtr out;
  auto check = std::make_pair(vp, Value::placeHolder);
  if (Value::cache.find(check) != Value::cache.end()) {
    // info("find a already exist Value");
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

  return out;
}

ValuePtr operator-(ValuePtr vp) {
  // std::lock_guard<std::mutex> lock(Value::mtx);

  double num = -vp->val;

  ValuePtr out;
  auto check = std::make_pair(vp, Value::placeHolder);
  if (Value::cache.find(check) != Value::cache.end()) {
    // info("find a already exist Value");
    out = Value::cache[check];
    out->val = num;
    out->derivative = 0;
  } else {
    out = std::make_shared<Value>(num);
    out->op = Operation::NEG;
    out->prev_.push_back(vp);

    Value::cache.insert({check, out});
  }
  return out;
}

ValuePtr operator-(ValuePtr lhs, ValuePtr rhs) {
  ValuePtr inter = -rhs;
  return lhs + inter;
}

ValuePtr inv(ValuePtr vp) {
  // std::lock_guard<std::mutex> lock(Value::mtx);

  double num = pow(vp->val, -1);

  ValuePtr out;
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
  return out;
}

ValuePtr operator/(ValuePtr lhs, ValuePtr rhs) {
  ValuePtr inter = inv(rhs);
  info("///////////////////////");
  // ValuePtr inter = pow(rhs,-1);

  return lhs * inter;
}

ValuePtr exp(ValuePtr vp) {
  // std::lock_guard<std::mutex> lock(Value::mtx);

  double num = exp(vp->val);

  ValuePtr out;
  auto check = std::make_pair(vp, Value::placeHolder);
  if (Value::cache.find(check) != Value::cache.end()) {
    // info("find a already exist Value");
    out = Value::cache[check];
    out->val = num;
    out->derivative = 0;
  } else {
    out = std::make_shared<Value>(num);
    out->op = Operation::EXP;
    out->prev_.push_back(vp);

    Value::cache.insert({check, out});
  }
  return out;
}

ValuePtr pow(ValuePtr lhs, ValuePtr rhs) {
  // std::lock_guard<std::mutex> lock(Value::mtx);

  double num = pow(lhs->val, rhs->val);

  ValuePtr out;
  auto check = std::make_pair(lhs, rhs);
  if (Value::cache.find(check) != Value::cache.end()) {
    // info("find a already exist Value");
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
  return out;
}

ValuePtr pow(ValuePtr lhs, double rhs) {
  ValuePtr rhsNum;
  {
    // std::lock_guard<std::mutex> lock(Value::mtx);

    double num = rhs;
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
  // std::lock_guard<std::mutex> lock(Value::mtx);

  double num;
  if (vp->val < 0) {
    num = 0.01 * vp->val;
  } else {
    num = vp->val;
  }

  ValuePtr out;
  auto check = std::make_pair(vp, Value::placeHolder);
  if (Value::cache.find(check) != Value::cache.end()) {
    // info("find a already exist Value");
    out = Value::cache[check];
    out->val = num;
    out->derivative = 0;
  } else {
    out = std::make_shared<Value>(num);  // 这里最开始写错了：声明了一个新的out
    out->op = Operation::RELU;
    out->prev_.push_back(vp);

    Value::cache.insert({check, out});
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
        this->prev_[0]->derivative += 0.01 * this->derivative;
      }
      break;
    case Operation::ADDI:
      this->prev_[0]->derivative += this->derivative;
      break;
    case Operation::MULI:
      this->prev_[0]->derivative += this->prev_[1]->val * this->derivative;
      break;
    default:
      warn("无效输入---------");
  }
}