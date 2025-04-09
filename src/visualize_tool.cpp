#include "visualize_tool.h"

using Logger::info;
using Logger::warn;

static std::vector<ValuePtr> alreadyVisited;

Agnode_t* drawDataNode(ValuePtr nodePtr, Agraph_t* g) {
  std::string temp = std::to_string(nodePtr->id);
  char name[temp.size() + 1];
  strcpy(name, temp.c_str());

  temp = nodePtr->toString();
  char label[temp.size() + 1];
  strcpy(label, temp.c_str());

  Agnode_t* node = agnode(g, name, TRUE);
  agsafeset(node, "shape", "box", "");
  agsafeset(node, "label", label, "");
  return node;
}

Agnode_t* drawOpNode(ValuePtr nodePtr, Agraph_t* g) {
  std::string tempStr;
  std::stringstream ss;
  ss << toString(nodePtr->op);
  tempStr = ss.str();
  char label[tempStr.size() + 1];
  strcpy(label, tempStr.c_str());

  ss << nodePtr->id;
  tempStr = ss.str();
  char name[tempStr.size() + 1];
  strcpy(name, tempStr.c_str());

  Agnode_t* node = agnode(g, name, TRUE);
  agsafeset(node, "label", label, "");
  return node;
}

void drawAllNodesEdgesRecursive(ValuePtr curNode, Agnode_t* curAgnode, Agraph_t* g) {
  if (std::find(alreadyVisited.begin(), alreadyVisited.end(), curNode) != alreadyVisited.end()) {
    return;
  }

  if (curNode->op == Operation::INVALID) {
    return;
  }

  Agnode_t* opNode = drawOpNode(curNode, g);
  agedge(g, opNode, curAgnode, NULL, TRUE);
  alreadyVisited.push_back(curNode);
  for (ValuePtr n : curNode->prev_) {
    Agnode_t* NodeN = drawDataNode(n, g);
    agedge(g, NodeN, opNode, NULL, TRUE);
    drawAllNodesEdgesRecursive(n, NodeN, g);
  }
}

void drawALLNodesEdges(std::vector<ValuePtr> allNodes,
                       std::vector<std::pair<ValuePtr, ValuePtr>> allEdges, Agraph_t* g) {
  for (ValuePtr node : allNodes) {
    drawDataNode(node, g);
    if (node->op != Operation::INVALID) {
      drawOpNode(node, g);

      std::stringstream ss;
      std::string tempStr;
      ss << toString(node->op) << node->id;
      tempStr = ss.str();
      char name1[tempStr.size() + 1];
      strcpy(name1, tempStr.c_str());

      ss.str("");  // ss.clear()是清除错误状态，不是清除内容
      ss << node->id;
      std::string temp = ss.str();
      char name2[temp.size() + 1];
      strcpy(name2, temp.c_str());

      Agnode_t* node1 = agnode(g, name1, 0);
      Agnode_t* node2 = agnode(g, name2, 0);
      agedge(g, node1, node2, NULL, TRUE);
    }
  }

  for (auto edge : allEdges) {
    std::string temp = std::to_string(edge.first->id);
    char name1[temp.size() + 1];
    strcpy(name1, temp.c_str());

    std::stringstream ss;
    std::string tempStr;
    ss << toString(edge.second->op) << edge.second->id;
    tempStr = ss.str();
    char name2[tempStr.size() + 1];
    strcpy(name2, tempStr.c_str());

    Agnode_t* node1 = agnode(g, name1, 0);
    Agnode_t* node2 = agnode(g, name2, 0);
    agedge(g, node1, node2, NULL, TRUE);
  }
}

void getAllNodesEdges(ValuePtr result, std::vector<ValuePtr>& allNodes,
                      std::vector<std::pair<ValuePtr, ValuePtr>>& allEdges) {
  std::vector<ValuePtr> readyNodes;
  readyNodes.push_back(result);
  while (!readyNodes.empty()) {
    ValuePtr cur = readyNodes[0];
    readyNodes.erase(readyNodes.begin());

    if (std::find(allNodes.begin(), allNodes.end(), cur) == allNodes.end()) {
      allNodes.push_back(cur);
      for (ValuePtr node : cur->prev_) {
        allEdges.push_back({node, cur});

        if (std::find(readyNodes.begin(), readyNodes.end(), node) == readyNodes.end()) {
          readyNodes.push_back(node);
        }
      }
    }
  }
}

void drawGraph(ValuePtr result, std::string name, GVC_t* gvc) {
  char cname[name.size() + 1];
  strcpy(cname, name.c_str());

  Agraph_t* graph = agopen(cname, Agdirected, NULL);
  agsafeset(graph, "rankdir", "LR", "");

  //上面是递归的方式绘制，下面是获得所有Nodes和edges一起绘制
  if (false) {
    Agnode_t* resAgnode = drawDataNode(result, graph);
    drawAllNodesEdgesRecursive(result, resAgnode, graph);
    alreadyVisited.clear();
  } else {
    std::vector<ValuePtr> allNodes;
    std::vector<std::pair<ValuePtr, ValuePtr>> allEdges;
    getAllNodesEdges(result, allNodes, allEdges);
    drawALLNodesEdges(allNodes, allEdges, graph);
  }

  gvLayout(gvc, graph, "dot");
  gvRenderFilename(gvc, graph, "png", (name + ".png").c_str());
}

void drawDivideGraph(MLP& mlp) {
  std::unique_ptr<Timer> td = std::make_unique<Timer>("time cost for draw");
  std::vector<std::vector<InputVal>> inputsForDraw;
  inputsForDraw.reserve(3600);

  std::mutex mtx;
  auto generateChunk = [&inputsForDraw, &mtx](int iStart, int iEnd) {
    std::vector<std::vector<InputVal>> localInputsForDraw;
    localInputsForDraw.reserve((iEnd - iStart) * 60);
    std::vector<InputVal> input;
    input.reserve(2);

    for (int iStep = iStart; iStep < iEnd; ++iStep) {
      double i = (iStep - 30) * 0.1;
      for (int jStep = 0; jStep < 60; ++jStep) {
        double j = (jStep - 30) * 0.1;
        input.clear();
        input.emplace_back(i);
        input.emplace_back(j);

        localInputsForDraw.push_back(std::move(input));
      }
    }
    std::lock_guard<std::mutex> lock(mtx);
    inputsForDraw.insert(inputsForDraw.end(), localInputsForDraw.begin(), localInputsForDraw.end());
  };
  std::thread t1(generateChunk, 0, 30);
  generateChunk(30, 60);
  t1.join();

  std::vector<std::vector<InputVal>> inputsForDraw1;
  std::vector<std::vector<InputVal>> inputsForDraw2;
  inputsForDraw1.reserve(inputsForDraw.size() / 2);
  inputsForDraw2.reserve(inputsForDraw.size() / 2);
  inputsForDraw1.insert(inputsForDraw1.end(), inputsForDraw.begin(),
                        inputsForDraw.begin() + inputsForDraw.size() / 2);
  inputsForDraw2.insert(inputsForDraw2.end(), inputsForDraw.begin() + inputsForDraw.size() / 2,
                        inputsForDraw.end());
  std::vector<std::vector<ValuePtr>> yForDraw1;
  std::vector<std::vector<ValuePtr>> yForDraw2;

  std::thread t3(computeOutput, std::ref(mlp), std::ref(inputsForDraw1), std::ref(yForDraw1));

  std::thread t4(computeOutput, std::ref(mlp), std::ref(inputsForDraw2), std::ref(yForDraw2));

  t3.join();
  t4.join();

  std::vector<std::vector<ValuePtr>> yForDraw;
  yForDraw.reserve(inputsForDraw.size());
  yForDraw.insert(yForDraw.end(), yForDraw1.begin(), yForDraw1.end());
  yForDraw.insert(yForDraw.end(), yForDraw2.begin(), yForDraw2.end());

  inputsForDraw1.clear();
  inputsForDraw2.clear();
  yForDraw1.clear();
  yForDraw2.clear();

  std::ofstream fileOut("../outPutForDraw.txt");

  fileOut << std::fixed << std::setprecision(2);
  for (size_t i = 0; i < inputsForDraw.size(); ++i) {
    fileOut << inputsForDraw[i][0].val << " " << inputsForDraw[i][1].val << " "
            << (yForDraw[i][0]->val > 0 ? 1 : 0) << "\n";
  }

  std::ifstream file("../inputData.txt");
  if (!file) {
    warn("无法打开文件: ", "../inputData.txt");
    return;
  }
  file.clear();
  file.seekg(0, std::ios::beg);
  fileOut << file.rdbuf();

  inputsForDraw.clear();

  yForDraw.clear();
}
