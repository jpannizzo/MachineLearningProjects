#import random for action sampling
#import random
#import vizdoom for game env
from  vizdoom import *
#import numpy for identity matrix
import numpy as np

#Import env base class
from gym import Env
#Import gym spaces
from gym.spaces import Discrete, Box
#import opencv for greyscaling
import cv2

# Import Matplotlib to show the impact of framestacking
from matplotlib import pyplot as plt

#Environment that can be called. All functions can be utilized except render. Use VizDooms built in renderer
class VizDoomGym(Env):
    #Start env function
    def __init__(self, render=False):
        #inherit from Env
        super().__init__()
        #Setup Game
        self.game = DoomGame()
        self.game.load_config('./github/ViZDoom/scenarios/basic.cfg')

        #Render frame logic
        if render == False:
            self.game.set_window_visible(False)
        else:
            self.game.set_window_visible(True)

        #start game
        self.game.init()

        #Create action/observation space
        self.observation_space = Box(low=0, high=255, shape=(100,160,1), dtype=np.uint8)
        self.action_space=Discrete(3)

    #this is how we take a step in the env
    def step(self, action):
        #Specify action and take step
        '''
        left = [1,0,0]
        right = [0,1,0]
        shoot = [0,0,1]
        '''
        actions = np.identity(3, dtype=np.uint8)
        reward = self.game.make_action(actions[action])

        #get return variables
        #if there is no current game state (game over screen) then return 0s so env does not crash
        if self.game.get_state():
            state = self.game.get_state().screen_buffer
            state = self.grayscale(state)
            ammo = self.game.get_state().game_variables[0]
            info = {"ammo":ammo}
        else:
            state = np.zeros(self.observation_space.shape)
            info = 0

        info = {"info": info}
        done = self.game.is_episode_finished()

        return state, reward, done, info

    #call to close down the game
    def close(self):
        self.game.close()

    #predefined in VizDoom. this is how to render the game or env
    def render():
        pass

    #grayscale the game frame and resize it
    def grayscale(self, observation):
        #recolor to grayscale
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        #resize image to make it smaller. reduces amount of pixels to process
        resize = cv2.resize(gray, (160,100), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (100,160,1))
        return state
        
    #restarts the game
    def reset(self):
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        return self.grayscale(state)

'''
#example random gamestate below
game = DoomGame()
game.load_config('./github/ViZDoom/scenarios/basic.cfg')
game.init()

#create actions
actions = np.identity(3, dtype=np.uint8)
episodes = 10
for episode in range(episodes):
    game.new_episode()
    while not game.is_episode_finished():
        #Get game state
        state = game.get_state()
        #Get the game image
        img = state.screen_buffer
        #Get the game variables - ammo
        info = state.game_variables
        #Take an action and get reward. Set up frame skip to give time to eval action
        reward = game.make_action(random.choice(actions),4)
        #print reward
        print('reward: ', reward)
    #print total result
    print('Result: ', game.get_total_reward())
    '''
