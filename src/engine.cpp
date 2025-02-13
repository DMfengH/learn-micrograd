#include "value.h"

int Node::maxID = 0;

// 这里为什么使用智能指针？
// 因为Node的成员中有指向其他Node的指针 ×
// 因为要支持+num操作，而由num创建的Node对象，需要延长生命周期；使用shared_ptr让out共享所有权就比较好。
// 另外，捕获的内容会变化，所以只能是引用捕获，或者是值捕获指针,而引用捕获会遇到局部变量销毁的问题，所以最好用指针。
NodePtr operator+(NodePtr lhs, NodePtr rhs){
  NodePtr out = std::make_shared<Node>(lhs->val + rhs->val);
  out->op = "+";
  out->prev_.insert(lhs);
  out->prev_.insert(rhs);
  out->backward_ = [lhs, rhs, out](){     // 按照引用捕获，就会有问题！！！
      lhs->derivative += out->derivative;
      rhs->derivative += out->derivative;
    };
  return out;
}

NodePtr operator*(NodePtr lhs, NodePtr rhs){
  NodePtr out = std::make_shared<Node>(lhs->val * rhs->val);
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
NodePtr operator+(NodePtr lhs, float num){
  NodePtr numNode = std::make_shared<Node>(num);
  return lhs + numNode;
}
