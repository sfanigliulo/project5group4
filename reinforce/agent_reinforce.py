import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()

        self.state_space = state_space #ok
        self.action_space = action_space #ok

        self.hidden = 64 #ok
        self.tanh = torch.nn.Tanh()

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)
        
        self.init_weights()

    def init_weights(self):
        for m in self.modules(): #modules are linear1, linear2, linear3
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """
            Actor
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)
        
        return normal_dist



class Agent(object):
    def __init__(self, policy, alpha=1, baseline=0,device='cpu'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
        #Adam is an optimization algorithm that can be used to update
        #network weights iterative based in training data.
        self.gamma = 0.99
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []
        self.alpha = alpha
        self.baseline = baseline


    def update_policy(self):
        # SOFIA - I commented .squeeze(-1) for action_log_probs because it gives me problem on the dimension of that tensor
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device)#.squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)

        #
        # TODO 2.2.a:
        #             - compute discounted returns
        #             - compute policy gradient loss function given actions and returns
        #             - compute gradients and step the optimizer
        #
        
        ## REINFORCE algorithm from https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63 -ELISA
        # I have some problems with the baseline, how to "insert" it - ELISA
        discounted_rewards = []
        #BASELINE, any value is fine as long as it generates faster convergence - FRA
        for t in range(len(rewards)): #to obtain discounted rewards
          Gt = 0 
          pw = 0
          for r in rewards[t:]:
            Gt = Gt + self.alpha*self.gamma**pw * r
            pw = pw + 1
            discounted_rewards.append(Gt - self.baseline)
        

        
        discounted_rewards = torch.tensor(discounted_rewards)
        #BASELINE through normalization - FRA
        # SOFIA - The normalization of the discounted_rewards causes some problem when running the code
        #discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std()) # normalize discounted rewards

        policy_gradient = [] 
        # SOFIA - Here there was "for action_log_probs, ..." but I think it was a grammar mistake and I changed it into "for action_log_prob, ..."
        for action_log_prob, Gt in zip(action_log_probs, discounted_rewards):
          policy_gradient.append(-action_log_prob * Gt)
          
        optimizer_reinforce = self.optimizer  
        optimizer_reinforce.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient.backward()
        optimizer_reinforce.step()
        
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []

        return        

    def get_action(self, state, evaluation=False):
        x = torch.from_numpy(state).float().to(self.train_device)

        # SOFIA - I modified from normal_dist = self.policy(x) to this because we need normal_dist and it doesn't work with the former command
        normal_dist = self.policy.forward(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob

    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)
