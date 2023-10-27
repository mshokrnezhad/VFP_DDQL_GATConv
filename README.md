# VFP_DDQL_GNN
A Virtual Function Placement (VFP) method using Double Deep Q-Learning (DDQL) powered by Graph Neural Network (GNN) layers. 

This repository implements the VNF placement agent described here (https://ieeexplore.ieee.org/abstract/document/10207694/) using GNN.

This method employs ReNet as its learning environment. For more details, check here: https://github.com/mshokrnezhad/ReNet

* IMPORTANT NOTE: GATConv returns a set of values for each input graph node. Therefore, it is extremely useful for methods attempting to comprehend the nodes based on their relationships in a graph. I require a GNN layer that returns values for the graph itself as a whole as opposed to its nodes. It may be possible via GATConv, but I do not have time for further investigation at this time.
