import re
import os
import numpy as np
import string
from ale_python_interface import ALEInterface
import gym

def create_env(config):
    if len(config.env_name.split("-")) == 1:
        # if env does not end in -v0 use alewrap
        class Env:
            def __init__(self):
                self.ale = ALEInterface()
                rom_name = "roms/Breakout.bin"
                self.ale.setInt("frame_skip", 4)
                self.ale.loadROM(rom_name)
                legal_actions = self.ale.getMinimalActionSet()
                self.action_map = {}
                for i in range(len(legal_actions)):
                    self.action_map[i] = legal_actions[i]
                self.action_num = len(self.action_map)

            def reset(self):
                state = np.zeros((84, 84, 3), dtype=np.uint8)
                self.ale.reset_game()
                return state

            def step(self, action):
                reward = self.ale.act(self.action_map[action])
                state = self.ale.getScreenRGB()
                done = self.ale.game_over()
                return state, reward, done, ""
        env = Env()
        config.action_num = env.action_num
    else:
        # else use gym
        env = gym.make(config.env_name)
        config.action_num = env.action_space.n
    return env


def load_checkpoint(sess, saver, config):
    run_num = re.search(r"\d+", config.load_checkpoint)
    if run_num == None:
        raise Exception("Not a run number in checkpoint file")
    run_num = run_num.group()
    for folder in os.listdir("log"):
        folder_num = re.search(r"\d+", folder)
        if folder_num and folder_num.group() == run_num:
            ckpt_file = "log/"+folder+"/checkpoint/"+config.load_checkpoint
    print("loading: " + ckpt_file)
    saver.restore(sess, ckpt_file)
