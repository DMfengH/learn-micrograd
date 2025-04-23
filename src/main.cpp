#include "test_ground.h"

int main() {
  testNN();

  // std::vector<ValuePtr> lhs;
  // std::vector<ValuePtr> rhs;
  // for (size_t i = 0; i < 3; ++i) {
  //   lhs.push_back(std::make_shared<Value>(2 * i, true));
  //   rhs.push_back(std::make_shared<Value>(2 * i + 1));
  // }

  // ValuePtr res = lhs * rhs;

  // GVC_t* gvc = gvContext();
  // std::string name1 = "MatMul";
  // drawGraph(res, name1, gvc);
  info(
      "---------------------------  The program is end at there  "
      "---------------------------");
}