#pragma once

#include "value.h"
#include "utilities.h"
#include "map"
#include <graphviz/cgraph.h>
#include <graphviz/gvc.h>

Agnode_t* drawDataNode(NodePtr nodePtr, Agraph_t* g);

Agnode_t* drawOpNode(NodePtr nodePtr, Agraph_t* g);

void draw(NodePtr curNode, Agnode_t* curAgnode, Agraph_t* g);

void drawGraph(NodePtr result, char* name, GVC_t* gvc);
