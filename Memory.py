import numpy as np
from torch_geometric.data import Data
import torch as T


class Memory(object):
    def __init__(self, MAX_SIZE, NUM_NODES, NUM_FEATURES):
         # Defining base variables
        self.MEMORY_SIZE = MAX_SIZE
        # Defining complementary variables
        self.counter = 0
        self.node_features_memory = np.zeros((self.MEMORY_SIZE, NUM_NODES, NUM_FEATURES), dtype=np.float32)
        self.resluted_node_features_memory = np.zeros((self.MEMORY_SIZE, NUM_NODES, NUM_FEATURES), dtype=np.float32)
        
        self.action_memory = np.zeros(self.MEMORY_SIZE, dtype=np.int64)
        self.reward_memory = np.zeros(self.MEMORY_SIZE, dtype=np.float32)
        self.terminal_memory = np.zeros(self.MEMORY_SIZE, dtype=np.bool)

    def store_transition(self, state, action, reward, resulted_state, done):    
        index = self.counter % self.MEMORY_SIZE

        self.node_features_memory[index] = state["NODE_FEATURES"]
        # self.link_matrix_memory[index] = state["LINK_MATRIX"]
        self.resluted_node_features_memory[index] = resulted_state["NODE_FEATURES"]
        # self.resulted_link_matrix_memory[index] = resulted_state["LINK_MATRIX"]
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.counter += 1

    def sample_buffer(self, batch_size):
        MAX_MEMORY_SIZE = min(self.counter, self.MEMORY_SIZE)
        batch = np.random.choice(MAX_MEMORY_SIZE, batch_size, replace=False)
        # print(batch)
        
        _node_features_memory = T.tensor(self.node_features_memory)
        _resluted_node_features_memory = T.tensor(self.resluted_node_features_memory)
        
        states = [Data(x=_node_features_memory[i]) for i in batch]     
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        resluted_states = [Data(x=_resluted_node_features_memory[i]) for i in batch]
        terminals = self.terminal_memory[batch]

        return states, actions, rewards, resluted_states, terminals
