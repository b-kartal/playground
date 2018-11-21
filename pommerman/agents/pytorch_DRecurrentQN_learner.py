# pytorch dqn implementation
# Input will be the observation of agents
# Currently, consider the fully observable scenario

# Modified from **Author**: `Adam Paszke <https://github.com/apaszke>`

import math
import random
from collections import namedtuple
import numpy as np
from itertools import count
import matplotlib
#import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from pommerman import constants


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'lstm_h', 'lstm_c')) # -  ``Transition`` - a named tuple representing a single transition in our environment -
                                                                                                  # LSTM hidden contains hx and cx - short and long memory

class ReplayMemory(object): #  a cyclic buffer of bounded size that holds the transitions observed recently.

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):              # returns a random batch of transititons for training
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DRecQN(nn.Module):

    def __init__(self):
        super(DRecQN, self).__init__()

        # current input is 18 x 11 x 11
        final_layer = {11: 7744, 8: 4096} # trial-error computed for this CNN
        print(final_layer[constants.BOARD_SIZE])

        if constants.BOARD_SIZE not in final_layer.keys():
            print("board size not supported :) ")


        self.cnn_last_dim = 64
        self.lin_input = self.cnn_last_dim * constants.BOARD_SIZE * constants.BOARD_SIZE # TODO change this to

        self.conv1 = nn.Conv2d(18, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, self.cnn_last_dim, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(self.cnn_last_dim)

        self.lstm_hidden_size = 256


        self.lstm = nn.LSTMCell(self.lin_input, self.lstm_hidden_size)

        self.head = nn.Linear(256, 6) #TODO feed 6 from pommerman environment variable directly

    def forward(self, x, hx, cx):


        #print(f" hx size is {hx.size()} and max is {hx.max()}")

        print(f"x size is {x.shape}")

        x = self.conv1(x)
        print(f"x size is {x.shape}")

        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        #print(x.size())

        x = x.view(-1, self.lin_input) # to fix the shape before fully connected layer

        #print(f" size after flattened is {x.size()}") # 64 x 4096 now - 64 is the batch

        hx, cx = self.lstm(x, (hx, cx)) # takes in CNN output and lstm hidden & cell states

        x = hx # output of lstm is newly updated hidden state - will be fed to output linear head

        return self.head(x.view(x.size(0), -1)), hx, cx # returning output 6 values for each action and lstm outputs (hidden state and cell state for lstm layer)

######################################################################
# Training loop
# ^^^^^^^^^^^^^
#
# Finally, the code for training our model.
#
# Here, you can find an ``optimize_model`` function that performs a
# single step of the optimization. It first samples a batch, concatenates
# all the tensors into a single one, computes :math:`Q(s_t, a_t)` and
# :math:`V(s_{t+1}) = \max_a Q(s_{t+1}, a)`, and combines them into our
# loss. By defition we set :math:`V(s) = 0` if :math:`s` is a terminal
# state. We also use a target network to compute :math:`V(s_{t+1})` for
# added stability. The target network has its weights kept frozen most of
# the time, but is updated with the policy network's weights every so often.
# This is usually a set number of steps but we shall use episodes for
# simplicity.
#

####################################################################

class d_rec_qn_learning(object):
    def __init__(self, *args, **kwargs):
        super(d_rec_qn_learning, self).__init__(*args, **kwargs)

        self.current_state = None
        self.action = None
        self.next_state= None
        self.reward = None
        self.steps_done = 0
        self.buffer_size = 100000

        self.BATCH_SIZE = 64 # TODO arbitrarily choosen just as other parameters
        self.GAMMA = 0.999
        self.EPS_START = 1.0
        self.EPS_END = 0.05

        self.learning_begin = 0

        self.running_loss = 0
        self.running_loss_tracker = [] # this will keep the batch loss over time

        self.EPS_DECAY = 1000000 # at 5x exploration will be 0.005
        self.TARGET_UPDATE = 100 # every 100 games

        self.interact_counter = 0
        self.OPTIMIZER_CALL_FREQ = 4 # AFTER EVERY 4 INTERACTION - UPDATE THE NN

        self.numberOfEpisodes = None
        self.episode_durations = None
        self.episode_rewards = None
        self.action_histogram = [] # TODO might be useful to show overtime which actions are taken more

        # if gpu is to be used
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DRecQN().to(self.device)
        self.target_net = DRecQN().to(self.device)

        self.empty_lstm_h = torch.zeros(self.BATCH_SIZE, self.policy_net.lstm_hidden_size).to(self.device)  # to pass to optimizer for target
        self.empty_lstm_c = torch.zeros(self.BATCH_SIZE, self.policy_net.lstm_hidden_size).to(self.device)  #  ""

        self.lstm_layer_hidden = None
        self.lstm_layer_cell = None
        self.reset_lstm_hidden()

        print(f"network arhitecture is {self.policy_net}")

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # freeze the weights

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(self.buffer_size)

    def reset_lstm_hidden(self): # This should be called upon after each episode ending
        self.lstm_layer_hidden = torch.zeros(1, self.policy_net.lstm_hidden_size).to(self.device)  # we need to keep track of these - reset hidden state at each episode end?
        self.lstm_layer_cell = torch.zeros(1, self.policy_net.lstm_hidden_size).to(self.device)  # not in the replay buffer though ...

    def optimize_model(self):
        #print(f" buffer size {len(self.memory)}")
        if len(self.memory) < max(self.BATCH_SIZE, self.learning_begin):
            #print(f"buffer {len(self.memory)} is not bigger than batch size")
            return

       # print(self.policy_net)

       # print("\n\n\n")

       # print(self.target_net)

        transitions = self.memory.sample(self.BATCH_SIZE)

        #print(f" sampled transitions are {transitions}")
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        #print(f"as batch {batch}")

        # TODO should final states be passed as None or as filtered channels as it is now

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.uint8)

        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        lstm_h_batch = torch.cat(batch.lstm_h) # additional lstm vectors parameters to pass to the learning
        lstm_c_batch = torch.cat(batch.lstm_c) # for long term memory cell - state

       # print(reward_batch)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken

        #print(f" action batch are {action_batch}")

        state_action_values = self.policy_net(state_batch, lstm_h_batch, lstm_c_batch)[0].gather(1, action_batch) # 1 x Batch for Q(s,a)
        #print(f" next state values are {state_action_values}")

        #print("policy net passed")

        # Compute V(s_{t+1}) for all next states.

        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states, self.empty_lstm_h[0:len(non_final_next_states),:], self.empty_lstm_c[0:len(non_final_next_states),:])[0].max(1)[0].detach() # TODO make sure V(S_{t}) = 0 for terminal states !!!
       # print(f" next state values are {next_state_values}")

        #print("target passed")

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss

        #loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        #print(f"loss is {loss}")

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)  # gradient clipping for stability during training?
        self.optimizer.step()

        # print statistics

        self.running_loss += loss.item()

        if self.steps_done % self.BATCH_SIZE == 0:
            print(f" steps done is {self.steps_done} and batch size is {self.BATCH_SIZE} current episode is ")
            print(f"loss for {self.steps_done/self.BATCH_SIZE} batch is {self.running_loss}")
            self.running_loss_tracker.append(self.running_loss)
            self.running_loss = 0

    def save_to_buffer_learn(self, currentEpisodeNumber): # do the conversions
        # self.action is set during the self.act() call
        # self.current_state is converted to tensor during self.act() call as well

        if self.next_state is not None: # set to None for the terminal states
            self.next_state = torch.from_numpy(self.next_state).unsqueeze(0).float().to(self.device) # TODO

        self.reward = torch.tensor([self.reward], device=self.device).float() # make reward torch conversion here

        self.memory.push(self.current_state, self.action, self.next_state, self.reward, self.lstm_layer_hidden, self.lstm_layer_cell) # place hidden and cell state of lstm layer here as well
        #print("saved transition to memory")

        self.interact_counter += 1
        if self.interact_counter % self.OPTIMIZER_CALL_FREQ == 0:
            self.optimize_model()
            self.interact_counter = 0
            #print(" optimized the model ")

        if currentEpisodeNumber % self.TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def act(self, action_space, testing): # will do the action-selection of dqn
        # this does the learning - agent will simply call this function
        # this also saves state and action pair to be pushed to memory buffer later

        self.current_state = torch.from_numpy(self.current_state).unsqueeze(0).float().to(self.device)
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold or testing is True:
            with torch.no_grad(): # what does this do - no gradient update?
                self.action, self.lstm_layer_hidden, self.lstm_layer_cell = self.policy_net(self.current_state, self.lstm_layer_hidden, self.lstm_layer_cell ) # .max(1)[1].view(1, 1) # TODO now NN model should return lstm hidden and cell states as well
                self.action = self.action.max(1)[1].view(1, 1) # to make sure it is index to max action as integer
                #print(f" values are {self.lstm_layer_cell}")

        else:
            with torch.no_grad(): # random action selection but need to get hidden states of lstm therefore go forward
                #print("dqn learn act choosing random")
                _, self.lstm_layer_hidden, self.lstm_layer_cell = self.policy_net(self.current_state, self.lstm_layer_hidden, self.lstm_layer_cell) # TODO now NN model should return lstm hidden and cell states as well

                self.action = torch.tensor([[random.randrange(action_space)]], device=self.device, dtype=torch.long)

        #print(f"choosing action {self.action.item()}")

        return self.action.item() # item() converts 1d tensor to python number

    def save_trained_model(self, training_episode_total):
        torch.save(self.target_net.state_dict(), 'target_net.pt')
        torch.save(self.policy_net.state_dict(), 'policy_net.pt')
        self.save_loss_and_rewards()
        # TODO should we save replay buffer as well ?

    def load_trained_model(self, moreTraining):
        # need to change model for evaluation mode if model will be used for inference only
        # because batch-norm layers are in default train mode

        print("trying to load the model")

        self.target_net.load_state_dict(torch.load('target_net.pt'))
        self.policy_net.load_state_dict(torch.load('policy_net.pt'))
        self.optimizer = optim.RMSprop(self.policy_net.parameters()) # TODO make sure about this - are lstm hidden state and cell state included in the updated param list

        print("loaded pretrained models")

        if (moreTraining == False):
            self.target_net.eval()
            self.policy_net.eval()
            print("Changed model to inference only mode")

    def save_loss_and_rewards(self):
        N = 10
        moving_rewards = np.convolve(self.episode_rewards, np.ones((N,)) / N, mode='valid')
        moving_durations = np.convolve(self.episode_durations, np.ones((N,)) / N, mode='valid').astype(int)
        filename = "drqn_summary"

        np.savetxt("_rm_" + filename + ".txt", moving_rewards)
        np.savetxt("_tm_" + filename + ".txt", moving_durations)

        np.savetxt("_loss_" + filename + ".txt", np.asarray(self.running_loss_tracker))

        with open("_r_" + filename + ".txt", 'w') as file:
            file.write(str(self.episode_rewards) + '\n')

        with open("_t_" + filename + ".txt", 'w') as file:
            file.write(str(self.episode_durations) + '\n')





    def plot_durations(self):
        # set up matplotlib

        is_ipython = 'inline' in matplotlib.get_backend()
        if is_ipython:
            from IPython import display

        plt.ion()
        plt.figure(1)
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(self.episode_rewards)

        plt.figure(2)
        plt.clf()
        plt.title('Loss...')
        plt.xlabel('Minibatch')
        plt.ylabel('Loss')
        plt.plot(self.running_loss_tracker)

        plt.figure(3)
        plt.clf()
        plt.title('Episode Length...')
        plt.xlabel('Episode')
        plt.ylabel('Step count')
        plt.plot(self.episode_durations)


        plt.pause(0.00001)  # pause a bit so that plots are updated
        if is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())