import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Bernoulli
from typing import Tuple
import gymnasium as gym
from gymnasium import spaces

device = torch.device(os.environ.get('CHARM_DEVICE', 'cuda:0' if torch.cuda.is_available() else 'cpu'))

class RolloutBuffer:
    def __init__(self):
        self.actions = []  # List of ([1, 11], [1, 43])
        self.states = []  # List of [43]
        self.logprobs = []  # List of ([1], [1, 43])
        self.rewards = []  # List of scalar
        self.state_values = []  # List of scalar
        self.is_terminals = []  # List of bool
        self.infos = []  # Track infos for violations and rates

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.infos[:]

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_max, temp):
        super().__init__()
        self.temp = temp  # Scalar
        self.action_max = action_max  # Scalar: 0.19952623149688797 W
        self.num_cus = action_dim[0][0]  # Scalar: 11
        self.association_action_dim = action_dim[1]  # (43, 11)

        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim[0], 512),  # [43*6] -> [512]
            nn.ReLU(),
            nn.Linear(512, 512),  # [512] -> [512]
            nn.ReLU(),
            nn.Linear(512, 512),  # [512] -> [512]
        )

        # Auxiliary value head for PPG.
        self.aux_value = nn.Linear(512, 1)  # [512] -> [1]

        # Deployment head for x_i (binary deployment decisions).
        self.deployment_head = nn.Linear(512, self.num_cus)  # [512] -> [11]
        
        # Association action head.
        self.association_action_logits = nn.Linear(512, self.association_action_dim[0] * self.association_action_dim[1])  # [512] -> [473]
        self.reshaped_association_action = nn.Unflatten(-1, self.association_action_dim)  # [473] -> [43, 11]
        self.association_action = nn.Softmax(dim=-1)  # [43, 11] -> [43, 11]

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def set_temp(self, temp):
        self.temp = temp  # Scalar

    def forward(self, state: torch.Tensor, training_epoch: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        print(f"Timestep {training_epoch}: state range: min={state.min().item()}, max={state.max().item()}")
        if torch.any(torch.isnan(state)) or torch.any(torch.isinf(state)):
            state = torch.nan_to_num(state, nan=0.0, posinf=20, neginf=-0.1)   #Should not have this nan here, env is already converting to min and max sinr.

        emb = self.shared_layers(state)
        if torch.any(torch.isnan(emb)):
            emb = torch.nan_to_num(emb, nan=0.5, posinf=1e5, neginf=-1e5)

        if training_epoch % 1 == 0:
            print(f"emb range: min={emb.min().item()}, max={emb.max().item()}")
            print(self.deployment_head.weight.min().item(), self.deployment_head.weight.max().item())

        deploy_logits = self.deployment_head(emb)
        if torch.any(torch.isnan(deploy_logits)) or torch.any(torch.isinf(deploy_logits)):
            print(f"WARNING: NaN/Inf in deploy_logits: min={deploy_logits.min().item()}, max={deploy_logits.max().item()}")
            deploy_logits = torch.nan_to_num(deploy_logits, nan=0.5, posinf=1e3, neginf=-1e3)

        # Preserve the December default (+/-2), but expose it for a controlled
        # ablation. The default bounds Bernoulli probabilities to
        # [sigmoid(-2), sigmoid(2)] = [0.119, 0.881], which can prevent a
        # highly reliable raw deployment policy even after convergence.
        deploy_logit_clamp = float(os.environ.get("CHARM_DEPLOY_LOGIT_CLAMP", "2.0"))
        deploy_logits = torch.clamp(
            deploy_logits, min=-deploy_logit_clamp, max=deploy_logit_clamp
        )
        association_action_emb = self.association_action_logits(emb)
        if torch.any(torch.isnan(association_action_emb)) or torch.any(torch.isinf(association_action_emb)):
            print(f"WARNING: NaN/Inf in association_action_logits: min={association_action_emb.min().item()}, max={association_action_emb.max().item()}")
            association_action_emb = torch.nan_to_num(association_action_emb, nan=0.5, posinf=1e3, neginf=-1e3)
        association_action_emb = torch.clamp(association_action_emb, min=-1e3, max=1e3)
        reshaped_association_action = self.reshaped_association_action(association_action_emb)

        if torch.any(torch.isnan(reshaped_association_action)) or torch.any(torch.isinf(reshaped_association_action)):
            print(f"WARNING: NaN/Inf in reshaped_association_action: min={reshaped_association_action.min().item()}, max={reshaped_association_action.max().item()}")
            reshaped_association_action = torch.nan_to_num(reshaped_association_action, nan=0.0, posinf=1e3, neginf=-1e3)        

        association_action_out = self.association_action(reshaped_association_action)

        aux_out = self.aux_value(emb)
        return deploy_logits, reshaped_association_action, aux_out, association_action_out

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init, action_max, temp, NRF=12):
        super().__init__()
        self.action_std = action_std_init  # Scalar: 0.25
        self.action_dim = action_dim  # ([11], (43, 11))
        self.association_action_dim = action_dim[1]  # (43, 11)
        self.device = device
        self.actor = Actor(state_dim, action_dim, action_max, temp)  # Actor
        self.critic = nn.Sequential(
            nn.Linear(state_dim[0], 512),  # [43*6] -> [512]
            nn.ReLU(),
            nn.Linear(512, 512),  # [512] -> [512]
            nn.ReLU(),
            nn.Linear(512, 1)  # [512] -> [1]
        )
        self.training_epoch = 0
        self.NRF = NRF
        self.temp = temp  # Initial temperature for deployment head (added here)

        for m in self.critic.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std  # Scalar

    def apply_association_mask(self, association_action_out, x_i):
        """
        Apply proper masking to association probabilities.
        Works for both single sample (act) and batch (evaluate).
        Ensures undeployed CUs have exactly zero probability.
        """
        batch_size = x_i.shape[0]

        # Create deployment mask: [batch_size, 43, 11]
        deployment_mask = x_i.unsqueeze(1).expand(-1, self.action_dim[1][0], -1).float()

        # Apply hard mask
        masked_association_action_out = association_action_out * deployment_mask

        # Compute row sums
        row_sums = masked_association_action_out.sum(dim=-1, keepdim=True)  # [batch_size, 43, 1]

        # If row_sum > 0, normalize; else (no deployed globally), fallback to uniform over all CUs to allow sampling (invalid, but env penalizes)
        num_cus = self.action_dim[0][0]  # 11
        # uniform_all = torch.full_like(masked_association_action_out, 1.0 / num_cus)
        masked_association_action_out = torch.where(
            row_sums > 0,
            masked_association_action_out / (row_sums + 1e-10),
            masked_association_action_out
        )

        # masked_association_action_out = torch.clamp(masked_association_action_out, min=0.0, max=1.0)
        masked_association_action_out = torch.nan_to_num(masked_association_action_out, nan=1e-10, posinf=1.0, neginf=1e-10)
        masked_association_action_out = torch.clamp(masked_association_action_out, min=1e-10, max=1.0)

        # # Final normalization
        # row_sums_final = masked_association_action_out.sum(dim=-1, keepdim=True)
        # masked_association_action_out = masked_association_action_out / (row_sums_final + 1e-12)

        # Verification
        if deployment_mask.sum() > 0 and self.training_epoch % 100 == 0:
            undeployed_mask = 1 - deployment_mask
            undeployed_probs = masked_association_action_out * undeployed_mask
            max_undeployed_prob = undeployed_probs.max().item()
            if max_undeployed_prob > 1e-8:
                print(f"ERROR: Undeployed CUs have probability: {max_undeployed_prob}")

        return masked_association_action_out


    def act(self, state: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        state = torch.unsqueeze(state, 0)  # [43*6] -> [1, 43*6]

        deploy_logits, reshaped_association_action, aux_value, association_action_out = self.actor(state, training_epoch=self.training_epoch)  # (torch.Tensor [1,11], torch.Tensor [1,43,11], torch.Tensor [1], torch.Tensor [1,43,11])

        # effective_temp = self.temp * self.action_std  # Use action_std to scale temp
        #print(f"temp in act: {self.temp}")
        # Sample x_i first
        deploy_logits = deploy_logits #/ self.temp
        deploy_dist = Bernoulli(logits=deploy_logits)  # [1, 11]
        x_i = deploy_dist.sample()  # [1, 11], binary 0 or 1


        x_i_log_prob = deploy_dist.log_prob(x_i).sum(dim=-1)  # [1]

    
        masked_association_action_out = self.apply_association_mask(association_action_out, x_i)   

        # Sample u (association_action)
        association_dist = Categorical(probs=masked_association_action_out)  # [1, 43, 11]
        association_action = association_dist.sample()  # [1, 43]
        association_logprob = association_dist.log_prob(association_action)  # [1, 43]

        # Minimal Moves NRF Projection (from current code)
        association_action_np = association_action.cpu().numpy().squeeze()  # [43]
        cu_loads = np.bincount(association_action_np, minlength=self.action_dim[1][1])  # [11]
        pre_projected_cu_loads = cu_loads.copy()

        association_action = torch.tensor(association_action_np, device=self.device).unsqueeze(0)  # [1, 43]
        ########association_logprob -= 1.5 * projection_count  # Increased penalty 1.0

        action_logprob = (x_i_log_prob, association_logprob)  # ([1], [1, 43])
        actions = (x_i, association_action, pre_projected_cu_loads)  # ([1, 11], [1, 43],[11])
        state_val = self.critic(state)  # [1, 1]
        self.training_epoch += 1
        return actions, action_logprob, state_val.detach().squeeze(), aux_value.detach().squeeze()

    def evaluate(self, state: torch.Tensor, action: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        x_i, association_action = action  # [batch_size, 11], [batch_size, 43]
        batch_size = state.shape[0]

        deploy_logits, reshaped_association_action, aux_value, association_action_out = self.actor(state, training_epoch=self.training_epoch)  # (torch.Tensor [batch_size,11], torch.Tensor [batch_size,43,11], torch.Tensor [batch_size,1], torch.Tensor [batch_size,43,11])

        deploy_logits = deploy_logits #/ self.temp
        deploy_dist = Bernoulli(logits=deploy_logits)  # [batch_size, 11]
        
        # # Mask association_action_out based on x_i
        # masked_association_action_out = association_action_out.clone()  # [batch_size, 43, 11]
        # for i in range(batch_size):
        #     for c in range(self.action_dim[1][1]):  # 11
        #         if x_i[i, c] == 0:
        #             masked_association_action_out[i, :, c] = 0.0  # Zero out probabilities for undeployed CUs
        # row_sums = masked_association_action_out.sum(dim=-1, keepdim=True)  # [batch_size, 43, 1]
        # masked_association_action_out = torch.where(row_sums > 0, masked_association_action_out / (row_sums + 1e-10), masked_association_action_out)  # Avoid div by zero        

        # masked_association_action_out = torch.nan_to_num(masked_association_action_out, nan=1e-10, posinf=1.0, neginf=1e-10)
        # masked_association_action_out = torch.clamp(masked_association_action_out, min=1e-10, max=1.0)

        masked_association_action_out = self.apply_association_mask(association_action_out, x_i)

        association_dist = Categorical(probs=masked_association_action_out)  # [batch_size, 43, 11]
        
        x_i_log_prob = deploy_dist.log_prob(x_i).sum(dim=-1)  # [batch_size]
        association_logprob = association_dist.log_prob(association_action)  # [batch_size, 43]

        # ###### Added this for consistency..
        # # Recompute projection_count for the given association_action
        # association_action_np = association_action.cpu().numpy()
        # cu_loads = np.bincount(association_action_np.reshape(-1), minlength=self.action_dim[1][1])  # [11]
        # active_cus = np.where(x_i.cpu().numpy().any(axis=0))[0]  # Active CUs
        # projection_count = 0
        # for cu in active_cus:
        #     current_load = cu_loads[cu]
        #     if current_load > self.NRF:
        #         excess = current_load - self.NRF
        #         projection_count += excess  # Approximate total excess (simplified)
        # association_logprob -= 1.5 * projection_count  # Apply same penalty

        deploy_entropy = deploy_dist.entropy().sum(dim=-1)  # [batch_size]
        association_entropy = association_dist.entropy()  # [batch_size, 43]

        action_logprob = (x_i_log_prob, association_logprob)  # ([batch_size], [batch_size, 43])
        dist_entropy = (deploy_entropy, association_entropy)  # ([batch_size], [batch_size, 43])
        state_val = self.critic(state)  # [batch_size, 1]
        return action_logprob, state_val, dist_entropy, aux_value

class JointPPG:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 noise_scale=1.0, action_std=0.25, action_max=0.19952623149688797,temp=2.0):
        self.action_std = action_std  # Initial standard deviation: 0.25
        self.noise_scale = noise_scale  # Exploration noise scale: 1.0
        self.gamma = gamma  # Discount factor
        self.eps_clip = eps_clip  # PPO clipping parameter
        self.K_epochs = K_epochs  # Number of PPO update epochs
        self.aux_training_epochs = 6  # Number of auxiliary training epochs for PPG
        self.ppg_repeat = 2  # PPG update frequency
        self.training_epoch = 0

        self.action_max = action_max  # Maximum power: 0.19952623149688797 W
        self.num_dus = state_dim[0]  # Number of DUs: 43
        self.device = device
        self.action_dim = action_dim


        # Hyperparameters for adaptive learning
        self.lambda_rate = 6.0 # tried with 1.0
        self.lambda_nrf = 1.0
        self.mu_rate = 6.0   # tried with 1.0
        self.mu_nrf = 1.0
        self.beta_deploy = 1.0
        self.beta_assoc = 2.0
        self.entropy_weight = float(os.environ.get("CHARM_ENTROPY_WEIGHT", "1e-5"))
        self.temp = temp 
        #############
        self.lambda_2 = 10.0   # make sure these match from env
        self.lambda_3 = 10.0   # make sure these match from env


        # Initialize buffers.
        self.buffer = RolloutBuffer()  # Buffer for PPO updates
        self.ppg_buffer = RolloutBuffer()  # Buffer for PPG auxiliary updates

        # Initialize policy networks.
        self.policy = ActorCritic(
            state_dim, action_dim, action_std, action_max, self.temp
        ).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99)

        self.policy_old = ActorCritic(
            state_dim, action_dim, action_std, action_max, self.temp
        ).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)

    def set_action_std(self, new_action_std):
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.policy.set_action_std(max(self.policy.action_std * action_std_decay_rate, min_action_std))
        self.policy_old.set_action_std(max(self.policy_old.action_std * action_std_decay_rate, min_action_std))

    def decay_temp(self):
        self.temp = max(0.5, min(4.0, self.temp * 0.995))
        self.policy.temp = self.temp
        self.policy_old.temp = self.temp
    
    
    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Input state: [43]
        state = torch.FloatTensor(state).to(device)  # [43]
        with torch.no_grad():
            actions, action_logprob, state_val, aux_value = self.policy_old.act(state)  # actions: ([1, 11], [1, 43]), etc.
        self.buffer.states.append(state)  # [43]
        self.buffer.actions.append(actions)  # ([1, 11], [1, 43])
        self.buffer.logprobs.append(action_logprob)  # ([1], [1, 43])
        self.buffer.state_values.append(state_val)  # Scalar
        x_i, association_action, cu_loads = actions  # [1, 11], [1, 43] ,[11]      
        return (
            x_i.cpu().numpy(),  # [1, 11]
            association_action.cpu().numpy(),  # [1, 43]   
            cu_loads         
        )

    def update(self):
        self.training_epoch += 1
        rewards = []
        discounted_reward = 0.0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0.0
            reward_tensor = torch.tensor(float(reward), dtype=torch.float32, device=self.device)
            if torch.isnan(reward_tensor) or torch.isinf(reward_tensor):
                print(f"WARNING: Invalid reward: {reward_tensor.item()}")
                reward_tensor = torch.tensor(0.0, device=self.device)
            discounted_reward = reward_tensor + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        rewards = torch.nan_to_num(rewards, nan=0.0, posinf=10000.0, neginf=-10000.0)
        if self.training_epoch % 100 == 0:
            print(f"Pre-clipped rewards: min={rewards.min().item()}, max={rewards.max().item()}")
        rewards = torch.clamp(rewards, min=-10000.0, max=10000.0)
        if self.training_epoch % 100 == 0:
            print(f"Post-clipped rewards: min={rewards.min().item()}, max={rewards.max().item()}")

        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_x_i = torch.squeeze(torch.stack([a[0] for a in self.buffer.actions], dim=0)).detach().to(self.device)
        old_association_actions = torch.squeeze(torch.stack([a[1] for a in self.buffer.actions], dim=0)).detach().to(self.device)
        old_actions = (old_x_i, old_association_actions)

        old_x_i_logprobs = torch.squeeze(torch.stack([a[0] for a in self.buffer.logprobs], dim=0)).detach().to(self.device)
        old_association_logprobs = torch.squeeze(torch.stack([a[1] for a in self.buffer.logprobs], dim=0)).detach().to(self.device)
        old_logprobs = (old_x_i_logprobs, old_association_logprobs)

        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)
        advantages = rewards - old_state_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        advantages = torch.nan_to_num(advantages, nan=0.0, posinf=1000.0, neginf=-1000.0)
        if self.training_epoch % 100 == 0:
            print(f"Advantages range: min={advantages.min().item()}, max={advantages.max().item()}")

        # rate_viol = np.mean([np.mean(info['rates'] < 6.0) for info in self.buffer.infos])
        # rate_viol = np.mean([info['rate_viol'] for info in self.buffer.infos])   # mean of the short fall of rate_viol for the entire buffer
        # nrf_viol_rate = np.mean([info['nrf_viol'] for info in self.buffer.infos])

        l2 = np.mean([np.sum(info['reward_components']['g2_constraints']) for info in self.buffer.infos])
        l3 = np.mean([np.sum(info['reward_components']['g3_constraints']) for info in self.buffer.infos])

        self.lambda_2 = max(0.0,min(15.0, self.lambda_2 - 0.3*l2))
        self.lambda_3 = max(0.0, min(15.0, self.lambda_3 - 0.3*l3))

        # self.entropy_weight *= (1 - 0.05 * nrf_viol_rate - 0.01 * rate_viol)
        # self.beta_deploy = max(0.5, min(50.0, self.beta_deploy * (1 + 0.05 * nrf_viol_rate + 0.1 * rate_viol)))
        # self.beta_assoc = max(0.05, min(5.0, self.beta_assoc * (1 + 0.05 * nrf_viol_rate + 0.1 * rate_viol)))

        

        # if rate_viol < 0.2 and nrf_viol_rate < 0.1 and self.training_epoch % 50 == 0:
        if self.training_epoch % 50 == 0:
            # self.beta_deploy *= 0.99
            # self.beta_assoc *= 0.99            
            self.entropy_weight = max(5e-10, self.entropy_weight * 0.98)

        # self.lambda_rate = max(0.05, min(6.0, self.lambda_rate + 0.5 * rate_viol))
        # self.lambda_nrf = max(0.05, min(3.0, self.lambda_nrf + 0.1 * nrf_viol_rate))
        # self.mu_rate = max(0.05, min(8.0, self.mu_rate + 0.5 * rate_viol))
        # self.mu_nrf = max(0.05, min(3.0, self.mu_nrf + 0.1 * nrf_viol_rate))



        # if rate_viol < 0.05 and nrf_viol_rate < 0.05:
        #     self.lambda_rate *= 0.995
        #     self.mu_rate /= 1.05
        #     self.lambda_nrf *= 0.995
        #     self.mu_nrf /= 1.05

        # lagr_loss = self.lambda_rate * rate_viol + self.lambda_nrf * nrf_viol_rate + (0.5 / self.mu_rate) * (rate_viol ** 2) + (0.5 / self.mu_nrf) * (nrf_viol_rate ** 2)

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy, aux_value = self.policy.evaluate(old_states, old_actions)
            x_i_logprobs, association_logprobs = logprobs
            deploy_entropy, association_entropy = dist_entropy

            # Add these before computing ratios:
            print(f"x_i_logprobs: min={x_i_logprobs.min().item():.3f}, max={x_i_logprobs.max().item():.3f}")
            print(f"old_x_i_logprobs: min={old_x_i_logprobs.min().item():.3f}, max={old_x_i_logprobs.max().item():.3f}")
            print(f"association_logprobs.sum(1): min={association_logprobs.sum(dim=1).min().item():.3f}, max={association_logprobs.sum(dim=1).max().item():.3f}")
            print(f"old_association_logprobs.sum(1): min={old_association_logprobs.sum(dim=1).min().item():.3f}, max={old_association_logprobs.sum(dim=1).max().item():.3f}")

            x_i_ratios = torch.exp(x_i_logprobs - old_x_i_logprobs.detach())
            # association_ratios = torch.exp(association_logprobs.sum(dim=1) - old_association_logprobs.sum(dim=1).detach())
            # association_ratios = torch.exp((association_logprobs.sum(dim=1) - old_association_logprobs.sum(dim=1).detach()) / 43)  # Divide by D=43 Number of DUs
            association_ratios = torch.exp(torch.clamp((association_logprobs.sum(dim=1) - old_association_logprobs.sum(dim=1).detach()) , -10, 10))
            ratios = x_i_ratios * association_ratios
            ratios = torch.clamp(ratios, min=1e-10, max=1e10)

            # Add these after computing individual ratios:
            print(f"x_i_ratios: min={x_i_ratios.min().item():.3f}, max={x_i_ratios.max().item():.3f}")
            print(f"association_ratios: min={association_ratios.min().item():.3f}, max={association_ratios.max().item():.3f}")

            print(f"Deploy entropy before norm: {deploy_entropy.mean().item()}")
            print(f"Assoc entropy before norm: {association_entropy.mean().item()}")

            if self.training_epoch % 100 == 0:
                print(f"Ratios range: min={ratios.min().item()}, max={ratios.max().item()}")

            # deploy_entropy_norm = deploy_entropy / (11 * torch.log(torch.tensor(2.0, device=device)))
            # assoc_entropy_norm = association_entropy.sum(dim=1) / (43 * torch.log(torch.tensor(11.0, device=device)))
            # dist_entropy_norm = self.beta_deploy * deploy_entropy_norm + self.beta_assoc * assoc_entropy_norm

            deploy_entropy_norm = deploy_entropy / (self.action_dim[1][1] * torch.log(torch.tensor(2.0, device=device)))
            assoc_entropy_norm = association_entropy.sum(dim=1) / (self.action_dim[1][0] * torch.log(torch.tensor(self.action_dim[1][1], device=device)))
            dist_entropy_norm = self.beta_deploy * deploy_entropy_norm + self.beta_assoc * assoc_entropy_norm


            # dist_entropy_norm = self.beta_deploy * deploy_entropy + self.beta_assoc * association_entropy.sum(dim=1)

            print(f"Deploy entropy after norm: {deploy_entropy_norm.mean().item()}")
            print(f"Assoc entropy after norm: {assoc_entropy_norm.mean().item()}")

            # target_dist_entropy_norm = 0.9
            # dist_entropy_adjust = torch.clamp(target_dist_entropy_norm - dist_entropy_norm.mean(), min=-0.6, max=0.6)            
            # entropy_loss = (self.entropy_weight + dist_entropy_adjust) * dist_entropy_norm.mean()

            # dist_entropy_adjust = torch.clamp(target_dist_entropy_norm - dist_entropy_norm.mean(), min=-0.6, max=0.6)            
            entropy_loss = self.entropy_weight * dist_entropy_norm.mean()

            print(f"dist_entropy_norm.mean(): {dist_entropy_norm.mean().item()}")
            print(f"Advantages: min={advantages.min():.3f}, max={advantages.max():.3f}")
            print(f"Ratios range: min={ratios.min().item()}, max={ratios.max().item()}")



            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            ppo_loss = -torch.min(surr1, surr2)
            if os.environ.get("CHARM_STANDARD_PPO", "0") == "1":
                # Standard clipped PPO objective. The legacy negative-advantage
                # branch can turn a positive loss into 5*A (negative), which
                # reverses/distorts the policy update.
                ppg_loss = ppo_loss
            else:
                ppg_loss = torch.where(
                    advantages < 0,
                    torch.minimum(ppo_loss, 5.0 * advantages),
                    ppo_loss,
                )

            state_values = state_values.squeeze()
            # loss = ppg_loss.mean() - entropy_loss - lagr_loss + 0.1 * self.MseLoss(state_values, rewards)
            loss = ppg_loss.mean() - entropy_loss  + 0.1 * self.MseLoss(state_values, rewards)

            # print(f"UPDATE epoch {self.training_epoch}: ratios[{ratios.min().item():.3f},{ratios.max().item():.3f}], ppg_loss={ppg_loss.mean().item():.3f}, lagr_loss={lagr_loss:.3f}, grad_norm={grad_norm.item():.3f}")


            if torch.any(torch.isnan(loss)):
                print(f"WARNING: NaN in loss: min={loss.item()}, max={loss.item()}")

            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            print(f"UPDATE epoch {self.training_epoch}: ratios[{ratios.min().item():.3f},{ratios.max().item():.3f}], ppg_loss={ppg_loss.mean().item():.3f}, grad_norm={grad_norm.item():.3f}")
            if self.training_epoch % 100 == 0:

                print(f"Gradient norm: {grad_norm.item()}")
            self.optimizer.step()

        self.ppg_buffer.states += self.buffer.states
        self.ppg_buffer.actions += self.buffer.actions
        self.ppg_buffer.state_values += self.buffer.state_values
        self.ppg_buffer.logprobs += self.buffer.logprobs

        if self.training_epoch % self.ppg_repeat == 0:
            old_states = torch.squeeze(torch.stack(self.ppg_buffer.states, dim=0)).detach().to(device)
            old_x_i = torch.squeeze(torch.stack([a[0] for a in self.ppg_buffer.actions], dim=0)).detach().to(device)
            old_association_actions = torch.squeeze(torch.stack([a[1] for a in self.ppg_buffer.actions], dim=0)).detach().to(device)
            old_actions = (old_x_i, old_association_actions)

            old_x_i_logprobs = torch.squeeze(torch.stack([a[0] for a in self.ppg_buffer.logprobs], dim=0)).detach().to(device)
            old_association_logprobs = torch.squeeze(torch.stack([a[1] for a in self.ppg_buffer.logprobs], dim=0)).detach().to(device)
            old_logprobs = (old_x_i_logprobs, old_association_logprobs)

            old_state_values = torch.squeeze(torch.stack(self.ppg_buffer.state_values, dim=0)).detach().to(device)

            for _ in range(self.aux_training_epochs):
                logprobs, state_values, dist_entropy, aux_value = self.policy.evaluate(old_states, old_actions)
                x_i_logprobs, association_logprobs = logprobs

                loss = self.MseLoss(aux_value.squeeze(), old_state_values)
                loss += 0.25 * self.kl_loss(x_i_logprobs, old_x_i_logprobs)
                loss += 0.25 * self.kl_loss(association_logprobs.sum(dim=1), old_association_logprobs.sum(dim=1))

                if torch.any(torch.isnan(loss)):
                    print(f"WARNING: NaN in aux loss: min={loss.item()}, max={loss.item()}")

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                if self.training_epoch % 100 == 0:
                    print(f"Gradient norm (aux): {grad_norm.item()}")
                self.optimizer.step()

            self.ppg_buffer.clear()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()




    def save(self, checkpoint_path, timestep, optimizer, training_epoch, lambda_rate, lambda_nrf, mu_rate, mu_nrf,
             beta_deploy, beta_assoc, entropy_weight):
        torch.save({
            'model_state_dict': self.policy_old.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'timestep': timestep,
            'training_epoch': training_epoch,
            'lambda_rate': lambda_rate,
            'lambda_nrf': lambda_nrf,
            'mu_rate': mu_rate,
            'mu_nrf': mu_nrf,
            'beta_deploy': beta_deploy,
            'beta_assoc': beta_assoc,
            'entropy_weight': entropy_weight,
        }, checkpoint_path, _use_new_zipfile_serialization=False)

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.policy_old.load_state_dict(checkpoint['model_state_dict'])
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_epoch = checkpoint['training_epoch']
        self.lambda_rate = checkpoint['lambda_rate']
        self.lambda_nrf = checkpoint['lambda_nrf']
        self.mu_rate = checkpoint['mu_rate']
        self.mu_nrf = checkpoint['mu_nrf']
        self.beta_deploy = checkpoint['beta_deploy']
        self.beta_assoc = checkpoint['beta_assoc']
        self.entropy_weight = checkpoint['entropy_weight']
        return checkpoint['timestep']
   
