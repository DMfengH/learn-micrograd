#pragma once

#include <graphviz/cgraph.h>
#include <graphviz/gvc.h>

#include <fstream>
#include <iomanip>
#include <thread>

#include "engine.h"
#include "map"
#include "micrograd.h"
#include "nn.h"
#include "utils.h"

Agnode_t* drawDataNode(ValuePtr nodePtr, Agraph_t* g);

Agnode_t* drawOpNode(ValuePtr nodePtr, Agraph_t* g);

void drawAllNodesEdgesRecursive(ValuePtr curNode, Agnode_t* curAgnode, Agraph_t* g);

void drawGraph(ValuePtr result, std::string name, GVC_t* gvc);

void drawDivideGraph(MLP& mlp);
