import numpy as np
import random

class ReplayMemory:
    def __init__(self, config):
        self.capacity = config.replay_memory_capacity
        self.batch_size = config.batch_size
        self.buff_size = config.buff_size
        self.screens = np.empty((self.capacity, 84,84), dtype=np.int8)
        self.actions = np.empty((self.capacity), dtype=np.int8)
        self.rewards = np.empty((self.capacity), dtype=np.int8)
        self.terminals = np.empty((self.capacity), dtype=np.bool)
        self.next_state_batch = np.empty((self.batch_size, 84, 84, 4), dtype=np.int8)
        self.state_batch = np.empty((self.batch_size, 84, 84, 4), dtype=np.int8)
        self.current = 0
        self.step = 0
        self.filled = False

    def add(self, screen, action, reward, terminal):
        self.screens[self.current] = screen
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.terminals[self.current] = terminal
        self.current += 1
        self.step += 1
        if self.current == self.capacity:
            self.current = 0
            self.filled = True

    def get_state(self, index):
        if self.filled == False:
            assert index < self.current, "%i index has note been added yet"%index
        #Fast slice read
        if index >= self.buff_size - 1:
            state = self.screens[(index - self.buff_size+1):(index + 1), ...]
        #Slow list read
        else:
            indexes = [(index - i) % self.step for i in reversed(range(self.buff_size))]
            state = self.screens[indexes, ...]
        # different screens should be in the 3rd dim as channels
        return np.transpose(state, [1,2,0])

    def sample_transition_batch(self):
        assert self.current >= self.batch_size, "There is not enough to sample. Call add bathc_size times"
        indexes = []
        while len(indexes) != self.batch_size:
            # index, is the index of state, and index + 1 of next_state
            if self.filled:
                index = random.randint(0, self.capacity - 2) # -2 because index +1 will be used for next state
                if index >= self.current and index - self.buff_size < self.current:
                    continue
            else:
                # can't start from 0 because get_state would loop back to the end -- wich is uninitialized.
                # index +1 can be terminal
                index = random.randint(self.buff_size -1, self.current - 2)

            if self.terminals[(index - self.buff_size):index].any():
                continue
            self.state_batch[len(indexes)] = self.get_state(index)
            self.next_state_batch[len(indexes)] = self.get_state(index + 1)
            indexes.append(index)
        action_batch = self.actions[indexes]
        reward_batch = self.rewards[indexes]
        terminal_batch = self.terminals[indexes]
        return self.state_batch, action_batch, reward_batch, self.next_state_batch, terminal_batch
