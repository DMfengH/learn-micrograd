#include "engine.h"

int Value::maxID = 0;

// 这里为什么使用智能指针？
// 因为Node的成员中有指向其他Node的指针 ×
// 新创建的Node对象，需要延长生命周期；使用shared_ptr让out共享所有权就比较好。
// 另外，捕获的内容会变化，所以只能是引用捕获，或者是值捕获指针,而引用捕获会遇到局部变量销毁的问题，所以最好用指针。
ValuePtr operator+(ValuePtr lhs, ValuePtr rhs){
  ValuePtr out = std::make_shared<Value>(lhs->val + rhs->val);
  out->op = "+";
  out->prev_.insert(lhs);
  out->prev_.insert(rhs);
  out->backward_ = [lhs, rhs, out](){     // 按照引用捕获，就会有问题！！！
      lhs->derivative += out->derivative;
      rhs->derivative += out->derivative;
    };
  return out;
}

ValuePtr operator*(ValuePtr lhs, ValuePtr rhs){
  ValuePtr out = std::make_shared<Value>(lhs->val * rhs->val);
  out->op = "*";
  out->prev_.insert(lhs);
  out->prev_.insert(rhs);
  out->backward_ = [lhs, rhs, out](){
      lhs->derivative += rhs->val * out->derivative;
      rhs->derivative += lhs->val * out->derivative;
    };
  return out;
}

// 这里这个num的所有权应该在哪里？？？最终是在它计算的out那里
// 为了把num传出去，只能使用智能指针
// 离开作用域还能用，就得是heap上的变量或者拷贝一份，方便管理所以用智能指针
ValuePtr operator+(ValuePtr lhs, double num){
  ValuePtr numNode = std::make_shared<Value>(num);
  return lhs + numNode;
}

ValuePtr operator*(ValuePtr lhs, double num){
  ValuePtr numNode = std::make_shared<Value>(num);
  return lhs * numNode;
}

ValuePtr tanh(ValuePtr vp){
  double num = (pow(M_E, 2*vp->val) -1) / (pow(M_E, 2*vp->val) +1);
  ValuePtr out = std::make_shared<Value>(num);
  out->op = "tanh";
  out->prev_.insert(vp);
  out->backward_ = [vp, out](){
      vp->derivative += (1-pow(out->val,2)) * out->derivative;
    };

  // auto inter1 = exp(vp);
  // auto inter2 = exp(-vp);
  // auto inter3 = inter1 - inter2;
  // auto inter4 = inter1 + inter2;
  // auto out = inter3 / inter4;

  return out;
}

ValuePtr operator-(ValuePtr vp){
  ValuePtr out = std::make_shared<Value>(-vp->val);
  out->op = "neg";
  out->prev_.insert(vp);
  out->backward_ = [vp, out](){
      vp->derivative += -1 * out->derivative;
    };
  return out;
}

ValuePtr operator-(ValuePtr lhs, ValuePtr rhs){
  ValuePtr inter = -rhs;
  return lhs + inter;
}

ValuePtr inv(ValuePtr vp){
  ValuePtr out = std::make_shared<Value>(pow(vp->val,-1));
  out->op = "inv";
  out->prev_.insert(vp);
  out->backward_ = [vp, out](){
      vp->derivative += -1 * pow(vp->val,-2) * out->derivative;
    };
  return out;
}

ValuePtr operator/(ValuePtr lhs, ValuePtr rhs){
  ValuePtr inter = inv(rhs);
  // ValuePtr inter = pow(rhs,-1);

  return lhs*inter;
}

ValuePtr exp(ValuePtr vp){
  ValuePtr out = std::make_shared<Value>(exp(vp->val));
  out->op = "exp";
  out->prev_.insert(vp);
  out->backward_ = [vp, out](){
      vp->derivative += out->val * out->derivative;
    };
  return out;
}

ValuePtr pow(ValuePtr lhs, ValuePtr rhs){
  ValuePtr out = std::make_shared<Value>(pow(lhs->val, rhs->val));
  out->op = "pow";
  out->prev_.insert(lhs);
  out->prev_.insert(rhs);
  out->backward_ = [lhs, rhs, out](){
      lhs->derivative += (rhs->val) * pow(lhs->val, rhs->val -1) * out->derivative;
      rhs->derivative += out->val * log(lhs->val) * out->derivative;
    };
  return out;
}

ValuePtr pow(ValuePtr lhs, double rhs){
  ValuePtr numLeaf = std::make_shared<Value>(rhs);
  return pow(lhs, numLeaf);
}