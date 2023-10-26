import numpy as np


class Memory(object):
    def __init__(self, MAX_SIZE, NODE_FEATURES_SHAPE, LINK_MATRIX_SHAPE):
         # Defining base variables
        self.MEMORY_SIZE = MAX_SIZE
        # Defining complementary variables
        self.counter = 0
        
        self.node_features_memory = np.zeros((self.MEMORY_SIZE, NODE_FEATURES_SHAPE[0], NODE_FEATURES_SHAPE[1]), dtype=np.float32)
        self.link_matrix_memory = np.zeros((self.MEMORY_SIZE, LINK_MATRIX_SHAPE[0], LINK_MATRIX_SHAPE[1]), dtype=np.float32)
        self.resluted_node_features_memory = np.zeros((self.MEMORY_SIZE, NODE_FEATURES_SHAPE[0], NODE_FEATURES_SHAPE[1]), dtype=np.float32)
        self.resulted_link_matrix_memory = np.zeros((self.MEMORY_SIZE, LINK_MATRIX_SHAPE[0], LINK_MATRIX_SHAPE[1]), dtype=np.float32)
        
        self.action_memory = np.zeros(self.MEMORY_SIZE, dtype=np.int64)
        self.reward_memory = np.zeros(self.MEMORY_SIZE, dtype=np.float32)
        self.terminal_memory = np.zeros(self.MEMORY_SIZE, dtype=np.bool)

    def store_transition(self, state, action, reward, resulted_state, done):    
        index = self.counter % self.MEMORY_SIZE

        self.node_features_memory[index] = state["NODE_FEATURES"]
        self.link_matrix_memory[index] = state["LINK_MATRIX"]
        self.resluted_node_features_memory[index] = resulted_state["NODE_FEATURES"]
        self.resulted_link_matrix_memory[index] = resulted_state["LINK_MATRIX"]
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.counter += 1

    def sample_buffer(self, batch_size):
        MAX_MEMORY_SIZE = min(self.counter, self.MEMORY_SIZE)
        batch = np.random.choice(MAX_MEMORY_SIZE, batch_size, replace=False)
        # print(batch)

        node_features = self.node_features_memory[batch]
        link_matrix = self.link_matrix_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        resluted_node_features = self.resluted_node_features_memory[batch]
        resluted_link_matrix = self.resulted_link_matrix_memory[batch]
        terminals = self.terminal_memory[batch]

        return node_features, link_matrix, actions, rewards, resluted_node_features,resluted_link_matrix, terminals
