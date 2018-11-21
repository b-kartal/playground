from pommerman.agents import BaseAgent
import numpy as np
import torch

from pommerman import constants
import torch.nn as nn
import torch.nn.functional as F
from pommerman.agents import dqn_agent_utilities as filter
from pommerman.agents import filter_action # Chao's rules

# provide input channels

N_S = None
N_A = None

CENTRALIZED_CRITIC = True
FILTER_RULES = True

def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        board_size = 11
        self.distribution = torch.distributions.Categorical
        self.a_dim = 6

        self.cnn_last_dim = 32
        self.post_cnn_layer_dim = 128

        self.lin_input = self.cnn_last_dim * board_size * board_size

        # Add LSTMCell here - see how the hidden state will be reset each turn
        # if LSTM_ENABLED:
        #    self.lstm = nn.LSTMCell(self.lin_input, self.post_cnn_layer_dim)
        # else:
        #    self.post_cnn_layer = nn.Linear(self.lin_input, self.post_cnn_layer_dim )
        #    set_init([self.post_cnn_layer])

        if CENTRALIZED_CRITIC: # TODO an LSTM can fit into this ....
            self.c_critic_conv1 = nn.Conv2d(20, 32, kernel_size=3, stride=1, padding=1) # TODO team single critic observation has 20 channels
            self.c_critic_conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
            self.c_critic_conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
            self.c_critic_conv4 = nn.Conv2d(32, self.cnn_last_dim, kernel_size=3, stride=1, padding=1)
            self.c_critic_ln1 = nn.Linear(self.lin_input, self.post_cnn_layer_dim)
            self.c_critic_ln2 = nn.Linear(self.post_cnn_layer_dim, self.post_cnn_layer_dim)
            self.c_critic_ln3 = nn.Linear(self.post_cnn_layer_dim, self.post_cnn_layer_dim)
            self.c_critic_head = nn.Linear(self.post_cnn_layer_dim, 1)

        self.conv1 = nn.Conv2d(19, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, self.cnn_last_dim, kernel_size=3, stride=1, padding=1)

        # Branch 1 for the first trained agent
        self.agent1_ln1 = nn.Linear(self.lin_input, self.post_cnn_layer_dim)
        self.agent1_ln2 = nn.Linear(self.post_cnn_layer_dim, self.post_cnn_layer_dim)
        self.agent1_ln3 = nn.Linear(self.post_cnn_layer_dim, self.post_cnn_layer_dim)
        self.agent1_head_actor = nn.Linear(self.post_cnn_layer_dim,self.a_dim)
        self.agent1_head_critic = nn.Linear(self.post_cnn_layer_dim,1)

        # Branch 2 for the second trained agent
        self.agent2_ln1 = nn.Linear(self.lin_input, self.post_cnn_layer_dim)
        self.agent2_ln2 = nn.Linear(self.post_cnn_layer_dim, self.post_cnn_layer_dim)
        self.agent2_ln3 = nn.Linear(self.post_cnn_layer_dim, self.post_cnn_layer_dim)
        self.agent2_head_actor = nn.Linear(self.post_cnn_layer_dim, self.a_dim)
        self.agent2_head_critic = nn.Linear(self.post_cnn_layer_dim, 1)

    def forward(self, x, agent_index=None, hx=None, cx=None): # pass agent id here to diverge on the branch - also pass ffa ...
        #TODO first teammate index can be 0 or 1 - second teammate index can be 2 or 3

        #print(f" size of hx is {hx.shape}")

        if CENTRALIZED_CRITIC and agent_index is None: # return centralized critic based on fully observable state information
            #print('passing here')
            y = F.elu(self.c_critic_conv1(x)) # TODO input here will be ffa state
            y = F.elu(self.c_critic_conv2(y))
            y = F.elu(self.c_critic_conv3(y))
            y = F.elu(self.c_critic_conv4(y))

            y = y.view(-1, self.lin_input)  # to fix the shape before fully connected layer

            y = F.elu(self.c_critic_ln1(y))
            y = F.elu(self.c_critic_ln2(y))
            y = F.elu(self.c_critic_ln3(y))
            return self.c_critic_head(y) # return a single central value estimate ...

        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))

        x = x.view(-1, self.lin_input)  # to fix the shape before fully connected layer

        # if LSTM_ENABLED:
        #    hx, cx = self.lstm(x, (hx,cx))
        #    x = hx
        # else:
        #    x = F.elu(self.post_cnn_layer(x))

        if agent_index in (0,1):                       # return first agent actor and LOCAL critic
            x = F.elu(self.agent1_ln1(x))
            x = F.elu(self.agent1_ln2(x))
            x = F.elu(self.agent1_ln3(x))
            agent1_logits = self.agent1_head_actor(x)
            agent1_values = self.agent1_head_critic(x)
            return agent1_logits, agent1_values, hx,cx
        else:                                          # return second agent actor and LOCAL critic
            x = F.elu(self.agent2_ln1(x))
            x = F.elu(self.agent2_ln2(x))
            x = F.elu(self.agent2_ln3(x))
            agent2_logits = self.agent2_head_actor(x)
            agent2_values = self.agent2_head_critic(x)
            return agent2_logits, agent2_values, hx, cx

    def choose_action(self, s, agent_index, hx=None, cx=None, value_viz_buffer=None, policy_viz_buffer=None):
        #print(s)
        # print(f"set to eval {s.shape}")
        self.eval()  # to freeze weights
        logits, value, hx, cx = self.forward(s, agent_index, hx, cx)

        prob = F.softmax(logits, dim=1).data
        #prob_np = prob.data.numpy()
        #print(f"probs are {prob_np}")
        #print(f"probs are {prob_np[0][0]}")

        if value_viz_buffer is not None: # TODO these parts have been added to log game data for visualization
            value_viz_buffer.append((value.data.numpy().flatten()))
        if policy_viz_buffer is not None:
            policy_viz_buffer.append((prob.data.numpy().flatten()))

        m = self.distribution(prob)

        #print(f"state value is {value}")

        return m.sample().numpy()[0], hx, cx, prob.data.numpy()[0] # also return probs


class bilal_ccritic_bignn_Alt_Agent(BaseAgent):

    def __init__(self, *args, **kwargs):
        super(bilal_ccritic_bignn_Alt_Agent, self).__init__(*args, **kwargs)
        self.gnet = Net()
        self.gnet.load_state_dict(torch.load('bilal_ccritic_bignn40K.pt'))
        self.game_step = 0

    def correct_nn_with_rules(self, obs, nn_intended_action, nn_probs):
        safe_actions = filter_action.get_filtered_actions(obs)
        if nn_intended_action not in safe_actions:
            nn_probs[np.setdiff1d([0, 1, 2, 3, 4, 5], safe_actions)] = 0.00000000000000000001 # added this due to 'fewer non-zero with p> 0 than size error'
            new_action = np.random.choice(6,1,replace=False,p=nn_probs/(sum(nn_probs)))
            if new_action is None:
                print("ERROR BY NONE Action")
            return new_action
        else:
            return nn_intended_action

    def act(self, obs, action_space):

        our_agent_pos = np.array(obs['position'])
        current_board = np.array(obs['board'])  # done
        #print('agent pos', our_agent_pos[0])
        #print('agent pos', our_agent_pos[1])

        our_agent_id = int(current_board[our_agent_pos[0]][our_agent_pos[1]])#
        #print('our agent id is ', our_agent_id)

        tm_filtered_state = v_wrap(filter.generate_NN_input_with_ids(our_agent_id, obs, self.game_step)).unsqueeze(0)  # TODO passed 10 as agent id - fix that ...
        tm_nn_action, _, _, NN_probs = self.gnet.choose_action(tm_filtered_state, our_agent_id-10)  # overload the first action
        if FILTER_RULES:
            tm_nn_action = self.correct_nn_with_rules(obs, tm_nn_action, NN_probs)

        self.game_step = self.game_step + 1

        return int(tm_nn_action)


    def episode_end(self, reward):
        self.game_step = 0
