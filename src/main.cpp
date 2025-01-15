#include "utilities.h"
#include "autoGrad.h"

#include <graphviz/cgraph.h>
#include <graphviz/gvc.h>

#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <memory>
#include <set>
#include <functional>
#include <bitset>

using Logger::info;
using Logger::warn;


#define BIT(x) (1<<x)

int main(){
  testAutoGrad();

  {
    // NodePtr a = std::make_shared<Node>('a',5);
    // NodePtr b = std::make_shared<Node>('b',2);
    // NodePtr c = a+b;
    // c->backward_();

  }
  





  info("The program is end at there");
}