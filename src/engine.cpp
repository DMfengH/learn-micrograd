#include "engine.h"

int Value::maxID = 0;
std::unordered_map<std::pair<ValuePtr, ValuePtr>, ValuePtr, PairHash> cache;
ValuePtr placeHolder = std::make_shared<Value>();

// 这里为什么使用智能指针？
// 因为Node的成员中有指向其他Node的指针 ×
// 新创建的Node对象，需要延长生命周期；使用shared_ptr让out共享所有权就比较好。
// 另外，捕获的内容会变化，所以只能是引用捕获，或者是值捕获指针,而引用捕获会遇到局部变量销毁的问题，所以最好用指针。
ValuePtr operator+(ValuePtr lhs, ValuePtr rhs){
  ValuePtr out;
  auto check = std::make_pair(lhs, rhs);

  if (cache.find(check) != cache.end()){
    // info("find a already exist Value");
    out = cache[check];
    out->val = lhs->val + rhs->val;
    out->derivative = 0;    // 在这里把所有梯度都设为0是不是一个好的时机？？？还是要专门写个函数把所有的Value的梯度都设置为0
  }else{
    out = std::make_shared<Value>(lhs->val + rhs->val);
    out->op = "+";
    out->derivative = 0;
    out->prev_.insert(lhs);
    out->prev_.insert(rhs);

    // std::weak_ptr 不能直接访问对象，它必须通过 lock() 转换成 std::shared_ptr，并且 lock() 可能失败（返回空指针 nullptr）。
    std::weak_ptr<Value> outWeak = out;
    out->backward_ = [lhs, rhs, outWeak](){
        if(auto outShared = outWeak.lock()){
          lhs->derivative += outShared->derivative;
          rhs->derivative += outShared->derivative;
        }
      };
    cache.insert({check, out});
  }
  
  return out;
}

ValuePtr operator*(ValuePtr lhs, ValuePtr rhs){
  ValuePtr out;
  auto check = std::make_pair(lhs, rhs);

  if (cache.find(check) != cache.end()){
    // info("find a already exist Value");
    out = cache[check];
    out->val = lhs->val * rhs->val;
    out->derivative = 0;
  }else{
    double a = lhs->val;
    double b = rhs->val;
    out = std::make_shared<Value>(a*b);
    out->op = "*";
    out->derivative = 0;
    out->prev_.insert(lhs);
    out->prev_.insert(rhs);

    std::weak_ptr<Value> outWeak = out;
    out->backward_ = [lhs, rhs, outWeak](){
        if(auto outShared = outWeak.lock()){
          lhs->derivative += rhs->val * outShared->derivative;
          rhs->derivative += lhs->val * outShared->derivative;
        }
      };
    cache.insert({check, out});
  }
  
  return out;
}

// 这里这个num的所有权应该在哪里？？？最终是在它计算的out那里
// 为了把num传出去，只能使用智能指针
// 离开作用域还能用，就得是heap上的变量或者拷贝一份，方便管理所以用智能指针
// 只支持和一个num相加，如何一个循环内lhs和多个num相加，会导致多个num提取到的都是同一个check
ValuePtr operator+(ValuePtr lhs, double num){
  ValuePtr rhsNum;
  auto check = std::make_pair(lhs, placeHolder);

  if (cache.find(check) != cache.end()){
    // info("find a already exist Value");
    rhsNum = cache[check];
    rhsNum->val = num;
  }else{
    rhsNum = std::make_shared<Value>(num);
    cache.insert({check, rhsNum});
  }
  return lhs + rhsNum;
}

ValuePtr operator*(ValuePtr lhs, double num){
  ValuePtr rhsNum;
  auto check = std::make_pair(lhs, placeHolder);

  if(cache.find(check) != cache.end()){
    // info("find a already exist Value");
    rhsNum = cache[check];
    rhsNum->val = num;
  }else{
    rhsNum = std::make_shared<Value>(num);
    cache.insert({check, rhsNum});
  }

  return lhs * rhsNum;
}

// tanh的计算有多种方式
// 每个value只能进行一个这种单操作数的操作
ValuePtr tanh(ValuePtr vp){
  double num = (pow(M_E, 2*vp->val) -1) / (pow(M_E, 2*vp->val) +1);

  ValuePtr out;
  auto check = std::make_pair(vp, placeHolder);
  if (cache.find(check) != cache.end()){
    // info("find a already exist Value");
    out = cache[check];
    out->val = num;
    out->derivative = 0;    
  }else{
    out = std::make_shared<Value>(num);
    out->op = "tanh";
    out->derivative = 0;  
    out->prev_.insert(vp);
    std::weak_ptr<Value> outWeak = out;
    out->backward_ = [vp, outWeak](){
      if(auto outShared = outWeak.lock()){
        vp->derivative += (1-pow(outShared->val,2)) * outShared->derivative;
      }
    };
    cache.insert({check, out});
  }

  return out;
}

ValuePtr operator-(ValuePtr vp){
  double num = -vp->val;

  ValuePtr out;
  auto check = std::make_pair(vp, placeHolder);
  if (cache.find(check) != cache.end()){
    // info("find a already exist Value");
    out = cache[check];
    out->val = num;
    out->derivative = 0;    
  }else{
    out = std::make_shared<Value>(num);
    out->op = "neg";
    out->prev_.insert(vp);
    std::weak_ptr<Value> outWeak = out;
    out->backward_ = [vp, outWeak](){
        if(auto outShared = outWeak.lock()){
          vp->derivative += -1 * outShared->derivative;
        }
      };
    cache.insert({check, out});
  }
  return out;
}

ValuePtr operator-(ValuePtr lhs, ValuePtr rhs){
  ValuePtr inter = -rhs;
  return lhs + inter;
}

ValuePtr inv(ValuePtr vp){
  double num = pow(vp->val,-1);

  ValuePtr out;
  auto check = std::make_pair(vp, placeHolder);
  if (cache.find(check) != cache.end()){
    // info("find a already exist Value");
    out = cache[check];
    out->val = num;
    out->derivative = 0;    
  }else{
    out = std::make_shared<Value>(num);
    out->op = "inv";
    out->prev_.insert(vp);
    std::weak_ptr<Value> outWeak = out;
    out->backward_ = [vp, outWeak](){
        if(auto outShared = outWeak.lock()){
          vp->derivative += -1 * pow(vp->val,-2) * outShared->derivative;
        }
      };
    cache.insert({check, out});
  }
  return out;
}

ValuePtr operator/(ValuePtr lhs, ValuePtr rhs){
  ValuePtr inter = inv(rhs);
  // ValuePtr inter = pow(rhs,-1);

  return lhs*inter;
}

ValuePtr exp(ValuePtr vp){
  double num = exp(vp->val);

  ValuePtr out;
  auto check = std::make_pair(vp, placeHolder);
  if (cache.find(check) != cache.end()){
    // info("find a already exist Value");
    out = cache[check];
    out->val = num;
    out->derivative = 0;    
  }else{
    out = std::make_shared<Value>(num);
    out->op = "exp";
    out->prev_.insert(vp);
    std::weak_ptr<Value> outWeak = out;
    out->backward_ = [vp, outWeak](){
        if(auto outShared = outWeak.lock()){
          vp->derivative += outShared->val * outShared->derivative;
        }
      };
    cache.insert({check, out});
  }
  return out;
}

ValuePtr pow(ValuePtr lhs, ValuePtr rhs){
  double num = pow(lhs->val, rhs->val);

  ValuePtr out;
  auto check = std::make_pair(lhs, rhs);
  if (cache.find(check) != cache.end()){
    // info("find a already exist Value");
    out = cache[check];
    out->val = num;
    out->derivative = 0;    
  }else{
    out = std::make_shared<Value>(num);
    out->op = "pow";
    out->prev_.insert(lhs);
    out->prev_.insert(rhs);
    std::weak_ptr<Value> outWeak = out;
    out->backward_ = [lhs, rhs, outWeak](){
        if(auto outShared = outWeak.lock()){
          lhs->derivative += (rhs->val) * pow(lhs->val, rhs->val -1) * outShared->derivative;
          rhs->derivative += outShared->val * log(lhs->val) * outShared->derivative;
        }
      };
    cache.insert({check, out});
  }
  return out;
}

ValuePtr pow(ValuePtr lhs, double rhs){
  double num = rhs;
  auto check = std::make_pair(lhs, placeHolder);
  
  ValuePtr rhsNum;
  if(cache.find(check) != cache.end()){
    rhsNum = cache[check];
    rhsNum->val = rhs;
  }else{
    rhsNum = std::make_shared<Value>(rhs);
    cache.insert({check,rhsNum});
  }

  return pow(lhs, rhsNum);
}

ValuePtr relu(ValuePtr vp){
  double num;
  if (vp->val < 0){
    num = 0;
  }else{
    num = vp->val;
  }
  
  ValuePtr out;
  auto check = std::make_pair(vp, placeHolder);
  if (cache.find(check) != cache.end()){
    // info("find a already exist Value");
    out = cache[check];
    out->val = num;
    out->derivative = 0;    
  }else{
    out = std::make_shared<Value>(num); // 这里最开始写错了：声明了一个新的out
    out->op = "relu";
    out->prev_.insert(vp);
    std::weak_ptr<Value> outWeak = out;
    out->backward_ = [vp, outWeak](){
        if(auto outShared = outWeak.lock()){
          if (outShared->val > 0){
            vp->derivative += outShared->derivative;
          }
        }
      };
    cache.insert({check, out});
  }
  return out;
}