#pragma once

#include "engine.h"
#include "utils.h"
#include "map"
#include <graphviz/cgraph.h>
#include <graphviz/gvc.h>

Agnode_t* drawDataNode(ValuePtr nodePtr, Agraph_t* g);

Agnode_t* drawOpNode(ValuePtr nodePtr, Agraph_t* g);

void drawAllNodesEdgesRecursive(ValuePtr curNode, Agnode_t* curAgnode, Agraph_t* g);

void drawGraph(ValuePtr result, char* name, GVC_t* gvc);
