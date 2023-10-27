import config
import numpy as np
import torch as T
from GNN import GNN
from Memory import Memory
rnd = np.random
AGENT_INIT = config.AGENT_INIT  # Default configs
from Functions import expand_array
from torch_geometric.data import Batch


class Agent(object):
    def __init__(self, INPUT):
        # Building INPUT
        self.INPUT = {key: INPUT[key] if key in INPUT else AGENT_INIT[key] for key in AGENT_INIT.keys()}
        # Defining base variables
        self.NAME = self.INPUT["NAME"]
        self.GAMMA = self.INPUT["GAMMA"]
        self.EPSILON = self.INPUT["EPSILON"]
        self.LR = self.INPUT["LR"]
        self.NUM_NODES = self.INPUT["NUM_NODES"]
        self.NUM_ACTIONS = self.INPUT["NUM_NODES"]
        self.NUM_FEATURES = self.INPUT["NUM_FEATURES"]
        self.LINKS_LIST = self.INPUT["LINKS_LIST"]
        self.BATCH_SIZE = self.INPUT["BATCH_SIZE"]
        self.MEMORY_SIZE = self.INPUT["MEMORY_SIZE"]
        self.EPSILON_MIN = self.INPUT["EPSILON_MIN"]
        self.EPSILON_DEC = self.INPUT["EPSILON_DEC"]
        self.REPLACE_COUNTER = self.INPUT["REPLACE_COUNTER"]
        self.CHECKPOINT_DIR = self.INPUT["CHECKPOINT_DIR"]
        self.ACTION_SPACE = [i for i in range(self.NUM_ACTIONS)]
        # Defining complementary variables
        self.learning_counter = 0
        self.edge_index = self.generate_edge_index()
        self.memory = Memory(MAX_SIZE=self.MEMORY_SIZE, NUM_NODES=self.NUM_NODES, NUM_FEATURES=self.NUM_FEATURES)
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
        states, actions, rewards, resluted_states, dones = self.memory.sample_buffer(self.BATCH_SIZE)
        rewards = T.tensor(rewards)
        actions = T.tensor(actions)
        dones = T.tensor(dones)

        return states, actions, rewards, resluted_states, dones

    def choose_action(self, state, SEED, train_mode=True):
        rnd.seed(SEED)
        if train_mode:
            random_number = rnd.random()
            # print(random_number)
            if random_number > self.EPSILON:
                state = T.tensor(state["NODE_FEATURES"], dtype=T.float)
                expected_values = self.q_eval.forward(state, self.edge_index)
                action = T.argmax(expected_values).item()
                # print("Q:", action)
            else:
                action = rnd.choice(self.ACTION_SPACE)
                # print("R:", action)
        else:
            state = T.tensor(state["NODE_FEATURES"], dtype=T.float)  # state = T.tensor([state], dtype=T.float)
            expected_values = self.q_eval.forward(state, self.edge_index)
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

    def generate_edge_index(self):
        edge_index = T.zeros(size=(2, self.NUM_FEATURES), dtype=T.int)
        
        for link in self.LINKS_LIST:
            i = self.LINKS_LIST.index(link)
            edge_index[0][i] = link[0]
            edge_index[1][i] = link[1]

        return edge_index
    
    def batch_learn(self):
        if self.memory.counter < self.BATCH_SIZE:
            return

        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()
        states, _actions, _rewards, resluted_states, _ = self.sample_memory()
        
        states_batch = Batch.from_data_list(states)
        x = states_batch.x
        actions = expand_array(_actions, self.NUM_NODES)
        rewards = expand_array(_rewards, self.NUM_NODES)
        resluted_states_batch = Batch.from_data_list(resluted_states)
        resluted_x = resluted_states_batch.x
        indexes = np.arange(self.BATCH_SIZE * self.NUM_NODES)
        
        _q_pred = self.q_eval.forward(x, self.edge_index)
        q_pred = _q_pred[indexes, actions]  # dims: batch_size * n_actions
        q_next = self.q_next.forward(resluted_x, self.edge_index)
        q_eval = self.q_eval.forward(resluted_x, self.edge_index)

        max_actions = T.argmax(q_eval, dim=1)
        # q_next[dones] = 0.0

        target = rewards + self.GAMMA * q_next[indexes, max_actions]

        loss = self.q_eval.criterion(target, q_pred)
        loss.backward()

        self.q_eval.optimizer.step()

        self.learning_counter += 1
        self.decrement_epsilon()

    def temporal_learn(self, _state, action, reward, _resulted_state):
            state = T.tensor(_state["NODE_FEATURES"], dtype=T.float)
            resulted_state = T.tensor(_resulted_state["NODE_FEATURES"], dtype=T.float)
            
            self.q_eval.optimizer.zero_grad()
            self.replace_target_network()
            
            _q_pred = self.q_eval.forward(state, self.edge_index)
            q_pred = _q_pred[action]  # dims: batch_size * n_actions
            q_next = self.q_next.forward(resulted_state, self.edge_index)
            q_eval = self.q_eval.forward(resulted_state, self.edge_index)

            max_action = T.argmax(q_eval, dim=1)
            # q_next[dones] = 0.0

            target = reward + self.GAMMA * q_next[max_action]

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
