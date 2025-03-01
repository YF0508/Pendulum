###########################################################################################
# Implementation of Deep Q-Learning Networks (DQN)
# Paper: https://www.nature.com/articles/nature14236
# Reference: https://github.com/Kchu/DeepRL_PyTorch
###########################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from replay_memory import ReplayBufferImage


'''DQN settings'''
# sequential images to define state
STATE_LEN = 4
# target policy sync interval
TARGET_REPLACE_ITER = 2
# (prioritized) experience replay memory size
MEMORY_CAPACITY = int(1e+5)
# gamma for MDP
GAMMA = 0.99

'''Training settings'''
# check GPU usage
USE_GPU = torch.cuda.is_available()
print('USE GPU: '+str(USE_GPU))
# mini-batch size
BATCH_SIZE = 64
# learning rate
LR = 2e-4
# the number of actions 
N_ACTIONS = 9
# the dimension of states
N_STATE = 4
# the multiple of tiling states 
N_TILE = 20


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        self.feature_extraction = nn.Sequential(
            # Conv2d(input channels, output channels, kernel_size, stride)
            nn.Conv2d(STATE_LEN, 8, kernel_size=8, stride=2),
            
            # TODO: ADD SUITABLE CNN LAYERS TO ACHIEVE BETTER PERFORMANCE
        )
           
        # action value
        self.fc_q = nn.Linear(8 * 8 * 8 + N_TILE * N_STATE, N_ACTIONS) 
        
        # TODO: ADD SUITABLE FULLY CONNECTED LAYERS TO ACHIEVE BETTER PERFORMANCE
        
        # initialization    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.orthogonal_(m.weight, gain = np.sqrt(2))
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            
    def forward(self, x, state):
        # x.size(0) : minibatch size
        mb_size = x.size(0)
        x = self.feature_extraction(x / 255.0) # (m, 9 * 9 * 10)
        x = x.view(x.size(0), -1)
        state = state.view(state.size(0), -1)
        state = torch.tile(state, (1, N_TILE))
        x = torch.cat((x, state), 1)
        x = F.relu(self.fc_0(x))
        x = F.relu(self.fc_1(x))
        action_value = self.fc_q(x)

        return action_value

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        

class DQN(object):
    def __init__(self):
        self.pred_net, self.target_net = ConvNet(), ConvNet()
        # sync target net
        self.update_target(self.target_net, self.pred_net, 1.0)
        # use gpu
        if USE_GPU:
            self.pred_net.cuda()
            self.target_net.cuda()
            
        # simulator step counter
        self.memory_counter = 0
        # target network step counter
        self.learn_step_counter = 0
        # loss function
        self.loss_function = nn.MSELoss()
        # create the replay buffer
        self.replay_buffer = ReplayBufferImage(MEMORY_CAPACITY)
        
        # define optimizer
        self.optimizer = torch.optim.Adam(self.pred_net.parameters(), lr=LR)
        
    def update_target(self, target, pred, update_rate):
        # update target network parameters using prediction network
        for target_param, pred_param in zip(target.parameters(), pred.parameters()):
            target_param.data.copy_((1.0 - update_rate) \
                                    * target_param.data + update_rate * pred_param.data)
    
    def save_model(self, pred_path, target_path):
        # save prediction network and target network
        self.pred_net.save(pred_path)
        self.target_net.save(target_path)

    def load_model(self, pred_path, target_path):
        # load prediction network and target network
        self.pred_net.load(pred_path)
        self.target_net.load(target_path)
        
    def save_buffer(self, buffer_path):
        self.replay_buffer.save_data(buffer_path)
        print("Successfully save buffer!")

    def load_buffer(self, buffer_path):
        # load data from the pkl file
        self.replay_buffer.read_list(buffer_path)

    def choose_action(self, s, epsilon, idling):
        
        # TODO: REPLACE THE FOLLOWING FAKE CODE WITH YOUR CODE
    
        action = np.random.randint(0, N_ACTIONS, 1)
        
        return action

    def store_transition(self, s, a, r, s_, done):
        self.memory_counter += 1
        self.replay_buffer.add(s, a, r, s_, float(done))

    def learn(self):
    
        # TODO: REPLACE THE FOLLOWING FAKE CODE WITH YOUR CODE
        
        loss = 1
        
        return loss
