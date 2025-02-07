#include "visualizeTool.h"

using Logger::info;
using Logger::warn;

static std::vector<NodePtr> alreadyVisited;

Agnode_t* drawDataNode(NodePtr nodePtr, Agraph_t* g){
  std::string temp = std::to_string(nodePtr->id);
  char name[temp.size()+1];
  strcpy(name, temp.c_str());
  
  temp = nodePtr->toString();
  char label[temp.size()+1];
  strcpy(label, temp.c_str());

  Agnode_t* node = agnode(g, name, TRUE);
  agsafeset(node, "shape", "box", "");
  agsafeset(node, "label",  label, "");    
  return node;
}

Agnode_t* drawOpNode(NodePtr nodePtr, Agraph_t* g){
  std::string tempStr;
  std::stringstream ss;
  ss << nodePtr->op;
  tempStr = ss.str(); 
  char label[tempStr.size()+1];
  strcpy(label, tempStr.c_str());

  ss << nodePtr->id;
  tempStr = ss.str();
  char name[tempStr.size()+1];
  strcpy(name, tempStr.c_str());

  Agnode_t* node = agnode(g, name, TRUE);
  agsafeset(node, "label", label, "");
  return node;
}

void draw(NodePtr curNode, Agnode_t* curAgnode, Agraph_t* g){
  if (std::find(alreadyVisited.begin(), alreadyVisited.end(), curNode) != alreadyVisited.end()){
    return;
  }

  if (curNode->op == ""){ return;}
  
  Agnode_t* opNode = drawOpNode(curNode, g);
  agedge(g, opNode, curAgnode, NULL, TRUE);
  alreadyVisited.push_back(curNode);
  for(NodePtr n:curNode->prev_){
    Agnode_t* NodeN = drawDataNode(n, g);
    agedge(g, NodeN, opNode, NULL, TRUE);
    draw(n, NodeN, g);
  }
}

void drawALLNodesEdges(std::vector<NodePtr> allNodes, std::vector<std::pair<NodePtr,NodePtr>> allEdges, Agraph_t* g){
  for (NodePtr node: allNodes){
    drawDataNode(node,g);
    if (!node->op.empty()){
      drawOpNode(node,g);

      std::stringstream ss;
      std::string tempStr;
      ss << node->op << node->id;
      tempStr = ss.str();
      char name1[tempStr.size()+1];
      strcpy(name1, tempStr.c_str()); 

      ss.str(""); // ss.clear()是清除错误状态，不是清除内容
      ss << node->id;
      std::string temp = ss.str();
      char name2[temp.size()+1];
      strcpy(name2, temp.c_str());

      Agnode_t *node1 = agnode(g, name1, 0);
      Agnode_t *node2 = agnode(g, name2, 0);
      agedge(g, node1, node2, NULL, TRUE);
      
    }
  }
  
  for(auto edge: allEdges){
    std::string temp = std::to_string(edge.first->id);
    char name1[temp.size()+1];
    strcpy(name1, temp.c_str());

    std::stringstream ss;
    std::string tempStr;
    ss << edge.second->op << edge.second->id;
    tempStr = ss.str();
    char name2[tempStr.size()+1];
    strcpy(name2, tempStr.c_str()); 

    Agnode_t *node1 = agnode(g, name1, 0);
    Agnode_t *node2 = agnode(g, name2, 0);
    agedge(g, node1, node2, NULL, TRUE);
  }
}

void getAllNodesEdges(NodePtr result, 
                      std::vector<NodePtr> &allNodes, 
                      std::vector<std::pair<NodePtr, NodePtr>> &allEdges){
  std::vector<NodePtr> readyNodes;
  readyNodes.push_back(result);
  while( !readyNodes.empty()){
    NodePtr cur = readyNodes[0];
    readyNodes.erase(readyNodes.begin());

    if(std::find(allNodes.begin(), allNodes.end(), cur) == allNodes.end()){
      allNodes.push_back(cur);
      for(NodePtr node: cur->prev_){
        allEdges.push_back({node, cur});

        if(std::find(readyNodes.begin(), readyNodes.end(), node) == readyNodes.end()){
          readyNodes.push_back(node);
        }
      }
    }
  }
}

void drawGraph(NodePtr result, char* name, GVC_t* gvc){
  Agraph_t* graph = agopen(name, Agdirected, NULL);
  agsafeset(graph, "rankdir", "LR", "");   

  //上面是递归的方式绘制，下面是获得所有Nodes和edges一起绘制
  if(false){
    Agnode_t* resAgnode = drawDataNode(result, graph);
    draw(result, resAgnode, graph);
    alreadyVisited.clear();
  }else{
    std::vector<NodePtr> allNodes;
    std::vector<std::pair<NodePtr, NodePtr>> allEdges;
    getAllNodesEdges(result, allNodes, allEdges);
    drawALLNodesEdges(allNodes, allEdges, graph);
  }
  
  gvLayout(gvc, graph, "dot");
  gvRenderFilename(gvc, graph, "png", (std::string(name) + ".png").c_str());
}
