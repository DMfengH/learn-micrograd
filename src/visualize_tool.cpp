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
