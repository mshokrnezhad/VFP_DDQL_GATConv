import config
import numpy as np
import torch as T
from GNN import GNN
from Memory import Memory
rnd = np.random
AGENT_INIT = config.AGENT_INIT  # Default configs


class Agent(object):
    def __init__(self, INPUT):
        # Building INPUT
        self.INPUT = {key: INPUT[key] if key in INPUT else AGENT_INIT[key] for key in AGENT_INIT.keys()}
        # Defining base variables
        self.NAME = self.INPUT["NAME"]
        self.GAMMA = self.INPUT["GAMMA"]
        self.EPSILON = self.INPUT["EPSILON"]
        self.LR = self.INPUT["LR"]
        self.NUM_ACTIONS = self.INPUT["NUM_ACTIONS"]
        self.NUM_FEATURES = self.INPUT["NUM_FEATURES"]
        self.NODE_FEATURES_SHAPE = self.INPUT["NODE_FEATURES_SHAPE"]
        self.LINK_MATRIX_SHAPE = self.INPUT["LINK_MATRIX_SHAPE"]
        self.BATCH_SIZE = self.INPUT["BATCH_SIZE"]
        self.MEMORY_SIZE = self.INPUT["MEMORY_SIZE"]
        self.EPSILON_MIN = self.INPUT["EPSILON_MIN"]
        self.EPSILON_DEC = self.INPUT["EPSILON_DEC"]
        self.REPLACE_COUNTER = self.INPUT["REPLACE_COUNTER"]
        self.CHECKPOINT_DIR = self.INPUT["CHECKPOINT_DIR"]
        self.ACTION_SPACE = [i for i in range(self.NUM_ACTIONS)]
        self.learning_counter = 0
        self.memory = Memory(MAX_SIZE=self.MEMORY_SIZE, NODE_FEATURES_SHAPE=self.NODE_FEATURES_SHAPE, LINK_MATRIX_SHAPE=self.LINK_MATRIX_SHAPE)
        self.q_eval = GNN(self.generate_q_inputs("_q_eval"))
        self.q_next = GNN(self.generate_q_inputs("_q_next"))

    def generate_q_inputs(self, local_name):
        SIZE_LAYERS = [self.NUM_FEATURES, self.NUM_FEATURES, self.NUM_FEATURES, self.NUM_ACTIONS]

        INPUT = {
            "LR": self.LR,
            "NAME": self.NAME + local_name,
            "CHECKPOINT_DIR": self.CHECKPOINT_DIR,
            "SIZE_LAYERS": SIZE_LAYERS
        }

        return INPUT

    def store_transition(self, state, action, reward, resulted_state, done):
        self.memory.store_transition(state, action, reward, resulted_state, done)

    def sample_memory(self):
        states_node_features, states_link_matrix, actions, rewards, resluted_states_node_features, resluted_states_link_matrix, dones = self.memory.sample_buffer(self.BATCH_SIZE)
        states_node_features = T.tensor(states_node_features)
        states_link_matrix = T.tensor(states_link_matrix)
        rewards = T.tensor(rewards)
        actions = T.tensor(actions)
        resluted_states_node_features = T.tensor(resluted_states_node_features)
        resluted_states_link_matrix = T.tensor(resluted_states_link_matrix)
        dones = T.tensor(dones)

        return states_node_features, states_link_matrix, actions, rewards, resluted_states_node_features, resluted_states_link_matrix, dones

    def choose_action(self, state, SEED, train_mode=True):
        rnd.seed(SEED)
        if train_mode:
            random_number = rnd.random()
            # print(random_number)
            if random_number > self.EPSILON:
                state = T.tensor(state, dtype=T.float)
                expected_values = self.q_eval.forward(state)
                action = T.argmax(expected_values).item()
                # print("Q:", action)
            else:
                action = rnd.choice(self.ACTION_SPACE)
                # print("R:", action)
        else:
            state = T.tensor(state, dtype=T.float)  # state = T.tensor([state], dtype=T.float)
            expected_values = self.q_eval.forward(state)
            action = T.argmax(expected_values).item()

        return action

    def replace_target_network(self):
        if self.learning_counter % self.REPLACE_COUNTER == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        if self.EPSILON > self.EPSILON_MIN:
            self.EPSILON = self.EPSILON - self.EPSILON_DEC
        else:
            self.EPSILON = self.EPSILON_MIN

    def learn(self):
        if self.memory.counter < self.BATCH_SIZE:
            return

        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()
        states_node_features, states_link_matrix, actions, rewards, resluted_states_node_features, resluted_states_link_matrix, dones    = self.sample_memory()
        indexes = np.arange(self.BATCH_SIZE)

        q_pred = self.q_eval.forward(states_node_features, states_link_matrix)[indexes, actions]  # dims: batch_size * n_actions
        q_next = self.q_next.forward(resluted_states_node_features, resluted_states_link_matrix)
        q_eval = self.q_eval.forward(resluted_states_node_features, resluted_states_link_matrix)

        max_actions = T.argmax(q_eval, dim=1)
        q_next[dones] = 0.0

        target = rewards + self.GAMMA * q_next[indexes, max_actions]

        loss = self.q_eval.criterion(target, q_pred)
        loss.backward()

        self.q_eval.optimizer.step()

        self.learning_counter += 1
        self.decrement_epsilon()

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def extract_model(self):
        self.q_eval.load_checkpoint()

        return self.q_eval
