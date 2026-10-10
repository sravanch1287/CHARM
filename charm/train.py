import os
if os.environ.get("CHARM_QUIET", "0") == "1":
    # The December code prints multiple large tensors per timestep; on Windows
    # this can dominate wall-clock time. Metrics still go to CSV/pickle/TB.
    import builtins
    builtins.print = lambda *_args, **_kwargs: None
import glob
import json
import time
import random
from datetime import datetime
import pickle
import torch
import numpy as np
import pandas as pd
from charm.ppg_policy import JointPPG

from charm.thz_environment import THzMidhaulEnv_SE_RL
from charm.physical_layer import Scenario
from charm.checkpoint import load_full_checkpoint, save_full_checkpoint


from torch.utils.tensorboard import SummaryWriter
from collections import Counter
import traceback


class NullSummaryWriter:
    """Logging-only fallback; deliberately has no effect on training state."""

    def add_scalar(self, *_args, **_kwargs):
        return None

    def close(self):
        return None


torch.set_num_threads(int(os.environ.get("CHARM_TORCH_THREADS", "1")))

# Environment parameters.  Defaults preserve every frozen reduced-scale run;
# explicit environment variables enable the reviewer-requested scalability
# experiment without maintaining a second trainer.
C = int(os.environ.get("CHARM_C", "5"))
D = int(os.environ.get("CHARM_D", "8"))
NRF = int(os.environ.get("CHARM_NRF", "3"))
if C <= 0 or D <= 0 or NRF <= 0:
    raise ValueError(f"C, D, and NRF must be positive; got C={C}, D={D}, NRF={NRF}")
if C * NRF < D:
    raise ValueError(
        f"Infeasible network dimensions: C*NRF={C * NRF} cannot serve D={D} DUs"
    )
PA = int(os.environ.get("CHARM_PA", "1"))
if PA not in (1, 2, 3, 4):
    raise ValueError(f"Unsupported CHARM_PA={PA}; expected 1=SLAPP, 2=DPA1, 3=DPA2, or 4=EPA")
PA_NAME = {1: "SLAPP", 2: "DPA1", 3: "DPA2", 4: "EPA"}[PA]
DATA_DIR = os.environ["CHARM_DATA_DIR"]
scn = Scenario(DATA_DIR)

def count_occurrences(arr, C):
    counts = Counter(arr)
    return [counts[i] for i in range(0, C)]  # Ensure order from 0 to C-1

def one_hot(a, N):
    s = np.zeros(N)
    s[a] = 1
    return s

def train():
    print("============================================================================================")
    print("SE-RL Training: With Adaptive Projection Penalties (Fix 3 ONLY)")
    city_name = 'boston'
    max_ep_len = int(os.environ.get("CHARM_EP_LEN", "80"))
    max_training_timesteps = int(int(os.environ.get("CHARM_MAX_EP", "800")) * max_ep_len)
    print_freq = max_ep_len * 2  # print avg reward in the interval
    log_freq = max_ep_len * 2  # log avg reward in the interval
    save_model_freq = int(os.environ.get("CHARM_SAVE_FREQ", "2000"))
    action_std = 0.25  # starting std for action distribution
    action_std_decay_rate = 0.00001  # linearly decay action_std
    min_action_std = 1e-6  # minimum action_std
    action_std_decay_freq = 300  # action_std decay frequency
    update_timestep = max_ep_len * 6  # update policy every n timesteps
    decay_temp_timestep = 200
    K_epochs = 30  # update policy for K epochs
    eps_clip = 0.15  # clip parameter for PPO
    gamma = 0.99  # discount factor
    lr_actor = 0.0001  # learning rate for actor network
    lr_critic = 0.0001  # learning rate for critic network
    random_seed = int(os.environ.get("CHARM_SEED", "0"))
    # Tuning runs require a real seed even when it is zero. The December code's
    # `if random_seed:` skipped seed 0 and was therefore unseeded.
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    print("City Name : " + city_name)
    env_name = "THzMidhaulEnv_SE_RL"
    print("training environment name : " + env_name)

    # Use SE-RL enhanced environment with projection penalties
    env = THzMidhaulEnv_SE_RL(C=C, D=D, scn=scn, PA=PA, NRF=NRF)
    print("PA : " + str(PA))

    print("============================================================================================")
    noise_scale = 1.0
    state_dim = (D * 6,)  # D*6 for flattened [D, 6] features
    actions_dim = [(C,), (D, C)]  # Reflects x_i, association
    action_max = env.max_p
    ppg_agent = JointPPG(
        state_dim, actions_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
        noise_scale=noise_scale, action_std=action_std, action_max=action_max, temp=2.0   #added temp=2.0 here
    )
    
    ppg_agent.lambda_2 = env.lambda_2    # making sure they are same
    ppg_agent.lambda_3 = env.lambda_3
    
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : " + str(start_time))
    print("============================================================================================")

    run_dir = os.environ["CHARM_RUN_DIR"]
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    if os.environ.get("CHARM_DISABLE_TENSORBOARD", "0") == "1":
        writer = NullSummaryWriter()
        print("TensorBoard disabled; pickle/checkpoint logging remains enabled.")
    else:
        writer = SummaryWriter(log_dir=log_dir)
    log_f_name = log_dir + '/training_log.csv'
    train_experiences_log_f_name = log_dir + "/experiences_train_log.pkl"

    run_num_pretrained = 0
    directory = os.path.join(run_dir, "checkpoints") + os.sep
    os.makedirs(directory, exist_ok=True)

    agent_checkpoint_path = directory + "Model_" + str(random_seed) + "_" + str(run_num_pretrained) + ".pth"
    print("PPG save checkpoint path : " + agent_checkpoint_path)

    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : " + str(max_training_timesteps))
    print("max timesteps per episode : " + str(max_ep_len))
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("Initializing a mixed discrete-continuous action space policy")
    print("starting std of action distribution : " + str(action_std))
    print("decay rate of std of action distribution : " + str(action_std_decay_rate))
    print("minimum std of action distribution : " + str(min_action_std))
    print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : " + str(K_epochs))
    print("PPO epsilon clip : " + str(eps_clip))
    print("discount factor (gamma) : " + str(gamma))
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : " + str(lr_actor))
    print("optimizer learning rate critic : " + str(lr_critic))
    print("setting random seed to " + str(random_seed))
    print("============================================================================================")

    # Training variables
    time_step = 0
    i_episode = 0

    # Logging lists - UPDATED TO INCLUDE NEW VARIABLES FROM ENHANCED ENV + SE-RL PROJECTION PENALTIES
    training_data = []
    episode_rewards = []
    episode_lengths = []
    episode_rate_violations = []
    episode_nrf_violations = []
    episode_qos_satisfactions = []
    episode_spectral_efficiencies = []
    episode_traffic_utilities = []
    episode_sum_rates = []              
    episode_deployment_costs = []       
    episode_main_rewards = []           
    episode_secondary_rewards = []      
    episode_constraint_penalties = []   
    episode_association_violations = []
    episode_load_balances = []
    episode_projection_stats = []           
    episode_projection_penalties_list = []
    
    # NEW: Additional diagnostic tracking for convergence monitoring
    convergence_history = []

    # Rolling averages - UPDATED TO INCLUDE NEW VARIABLES + SE-RL PROJECTION PENALTIES
    window_size = 100
    avg_reward = 0
    avg_ep_len = 0
    avg_rate_viol = 0
    avg_nrf_viol = 0
    avg_qos_satisfaction = 0
    avg_spectral_efficiency = 0
    avg_traffic_utility = 0
    avg_sum_rates = 0                   
    avg_deployment_cost = 0             
    avg_main_reward = 0                 
    avg_secondary_reward = 0            
    avg_constraint_penalties = 0        
    avg_association_viol = 0
    avg_load_balance = 0
    avg_projection_penalty = 0          
    print_running_reward = 0
    print_running_episodes = 0
    log_running_reward = 0
    log_running_episodes = 0
    CU_usage = np.zeros(C)

    full_checkpoint_freq = int(os.environ.get("CHARM_FULL_CHECKPOINT_FREQ", "25"))
    resume_checkpoint_path = os.environ.get("CHARM_RESUME_CHECKPOINT", "").strip()
    best_state = {
        "checkpoint": None,
        "completed_episodes": 0,
        "timestep": 0,
        "rolling_convergence": float("-inf"),
        "rolling_main_objective": float("-inf"),
    }
    checkpoint_run_config = {
        "seed": random_seed,
        "pa": PA,
        "episode_length": max_ep_len,
        "standard_ppo": os.environ.get("CHARM_STANDARD_PPO", "0"),
        "raw_nrf_lagrangian": os.environ.get("CHARM_RAW_NRF_LAGRANGIAN", "0"),
        "deploy_logit_clamp": os.environ.get("CHARM_DEPLOY_LOGIT_CLAMP", "2.0"),
        "post200_lr_factor": os.environ.get("CHARM_POST200_LR_FACTOR", "1.0"),
        "penalty_schedule": os.environ.get("CHARM_PENALTY_SCHEDULE", "conditional"),
        "base_safety": os.environ.get("CHARM_BASE_SAFETY", "50.0"),
        "base_nrf": os.environ.get("CHARM_BASE_NRF", "40.0"),
        "base_validity": os.environ.get("CHARM_BASE_VALIDITY", "20.0"),
        "base_movement": os.environ.get("CHARM_BASE_MOVEMENT", "10.0"),
    }

    if resume_checkpoint_path:
        resumed = load_full_checkpoint(
            resume_checkpoint_path, agent=ppg_agent, env=env
        )
        saved_run_config = dict(resumed["run_config"])
        # Full checkpoints written before PA selection became configurable are
        # necessarily SLAPP checkpoints because the trainer hardcoded PA=1.
        saved_run_config.setdefault("pa", 1)
        if saved_run_config != checkpoint_run_config:
            raise ValueError(
                "Resume configuration differs from the saved full checkpoint: "
                f"saved={saved_run_config}, current={checkpoint_run_config}"
            )
        time_step = int(resumed["timestep"])
        i_episode = int(resumed["next_episode"])
        restored = resumed["histories"]
        training_data = restored["training_data"]
        episode_rewards = restored["episode_rewards"]
        episode_lengths = restored["episode_lengths"]
        episode_rate_violations = restored["episode_rate_violations"]
        episode_nrf_violations = restored["episode_nrf_violations"]
        episode_qos_satisfactions = restored["episode_qos_satisfactions"]
        episode_spectral_efficiencies = restored["episode_spectral_efficiencies"]
        episode_traffic_utilities = restored["episode_traffic_utilities"]
        episode_sum_rates = restored["episode_sum_rates"]
        episode_deployment_costs = restored["episode_deployment_costs"]
        episode_main_rewards = restored["episode_main_rewards"]
        episode_secondary_rewards = restored["episode_secondary_rewards"]
        episode_constraint_penalties = restored["episode_constraint_penalties"]
        episode_association_violations = restored["episode_association_violations"]
        episode_load_balances = restored["episode_load_balances"]
        episode_projection_stats = restored["episode_projection_stats"]
        episode_projection_penalties_list = restored["episode_projection_penalties_list"]
        convergence_history = restored["convergence_history"]
        CU_usage = restored["CU_usage"]
        best_state = resumed["best_state"]
        print(
            f"Resumed exact episode-boundary state from {resume_checkpoint_path}; "
            f"next episode={i_episode}, timestep={time_step}"
        )

    print("Starting SE-RL enhanced training loop with Adaptive Projection Penalties...")
    print("============================================================================================")

    

    while time_step <= max_training_timesteps:
        state = env.reset()
        env.update_demands()   # To create demands until next update 
        current_ep_reward = 0.0
        current_ep_sum_rate = 0.0
        current_ep_length = 0
        current_ep_main_obj = 0.0
        current_ep_qos_satisfaction = 0.0
        current_ep_secondary_reward = 0.0
        ep_rate_violations = []
        ep_nrf_violations = []
        ep_qos_satisfactions = []
        ep_spectral_efficiencies = []
        ep_traffic_utilities = []
        ep_sum_rates = []                   
        ep_deployment_costs = []            
        ep_main_rewards = []                
        ep_secondary_rewards = []           
        ep_constraint_penalties = []        
        ep_association_violations = []
        ep_load_balances = []
        ep_projection_stats = []            
        episode_projection_penalties = 0.0  
        
        # NEW: Episode-level diagnostic tracking lists
        ep_raw_cus = []
        ep_projected_cus = []
        ep_cus_gap = []
        ep_penalty_obj_ratios = []
        ep_deployment_diversity = []
        ep_safety_success = []
        ep_validity_success = []
        ep_nrf_success = []

        start_time_eps = datetime.now().replace(microsecond=0)

        for t in range(1, max_ep_len + 1):
            x_i, association_action, cu_loads = ppg_agent.select_action(state)
            association_action = association_action.flatten()

            print("CUs deployed by RL: " + str(x_i))
            print("Association action: " + str(association_action))
            print("CUs selected by RL: " + str(np.unique(association_action)))
            CU_usage += x_i.squeeze(0)
            print("CU_Usage_freq : " + str(CU_usage / (time_step + 1)))
            print("Associated DUs per CU before projection: " + str(cu_loads))

            action = (x_i, association_action, cu_loads)
            state, reward, done_env, info = env.step(action)

            # ============= NEW: COMPREHENSIVE DIAGNOSTIC TRACKING =============
            
            # Extract raw and projected actions
            x_i_raw = info['raw_actions']['x_i_raw']
            x_i_projected = info['projected_actions']['x_i_projected']
            projection_info = info['projection_info']
            
            # 1. Raw vs Projected CUs (CRITICAL!)
            num_raw_cus = np.sum(x_i_raw)
            num_projected_cus = np.sum(x_i_projected)
            cus_gap = num_projected_cus - num_raw_cus
            writer.add_scalar('Deployment/raw_cus', num_raw_cus, time_step + 1)
            writer.add_scalar('Deployment/projected_cus', num_projected_cus, time_step + 1)
            writer.add_scalar('Deployment/cus_gap', cus_gap, time_step + 1)
            
            # 2. Penalty/Objective Ratio (CRITICAL!)
            main_obj = info['reward_components']['main_objective']
            proj_penalty = info['reward_components']['projection_penalties']
            penalty_obj_ratio = abs(proj_penalty) / (abs(main_obj) + 1e-8)
            writer.add_scalar('Diagnostics/penalty_objective_ratio', penalty_obj_ratio, time_step + 1)
            
            # 3. Deployment Diversity (Exploration indicator)
            unique_deployments = len(np.unique(x_i_raw))
            deployment_diversity = unique_deployments / float(C)
            writer.add_scalar('Exploration/deployment_diversity', deployment_diversity, time_step + 1)
            
            # 4. Per-timestep Success Flags (0 violations = success)
            safety_success = 1 if projection_info['safety_deployments_added'] == 0 else 0
            validity_success = 1 if projection_info['validity_reassignments'] == 0 else 0
            nrf_success = 1 if projection_info['nrf_violations_fixed'] == 0 else 0
            writer.add_scalar('Success_Timestep/safety_success', safety_success, time_step + 1)
            writer.add_scalar('Success_Timestep/validity_success', validity_success, time_step + 1)
            writer.add_scalar('Success_Timestep/nrf_success', nrf_success, time_step + 1)
            writer.add_scalar('Success_Timestep/overall_success', 
                            safety_success * validity_success * nrf_success, time_step + 1)
            
            # Collect for episode-level aggregation
            ep_raw_cus.append(num_raw_cus)
            ep_projected_cus.append(num_projected_cus)
            ep_cus_gap.append(cus_gap)
            ep_penalty_obj_ratios.append(penalty_obj_ratio)
            ep_deployment_diversity.append(deployment_diversity)
            ep_safety_success.append(safety_success)
            ep_validity_success.append(validity_success)
            ep_nrf_success.append(nrf_success)
            
            # ==================================================================

            current_ep_reward += reward
            current_ep_main_obj += info['reward_components']['main_objective']
            current_ep_qos_satisfaction += info['qos_satisfaction']
            current_ep_length += 1

            current_ep_sum_rate += np.sum(env.rate_list)

            print(f"Raw associations: {info['raw_actions']['raw_cu_loads']}")
            print(f"Projected associations: {info['projected_actions']['projected_cu_loads']}")

            # Collect episode statistics
            ep_rate_violations.append(info['rate_viol'])
            ep_nrf_violations.append(info['nrf_viol'])
            ep_qos_satisfactions.append(info['qos_satisfaction'])
            ep_spectral_efficiencies.append(info['spectral_efficiency'])
            ep_sum_rates.append(info['reward_components']['sum_rates'])                         
            ep_deployment_costs.append(info['reward_components']['deployment_cost'])           
            ep_main_rewards.append(info['reward_components']['main_objective'])                
            ep_association_violations.append(info['reward_components']['association_violations'])

            ep_projection_stats.append(projection_info)
            episode_projection_penalties += info['reward_components']['projection_penalties']

            # Store experience
            ppg_agent.buffer.rewards.append(reward)
            ppg_agent.buffer.is_terminals.append(done_env)
            ppg_agent.buffer.infos.append(info)

            print("Reward: " + str(reward))

            print("++++++++++++++")
            writer.add_scalar('Deployed_CUs_per_timestep/ncus/', np.sum(x_i), time_step + 1)
            writer.add_scalar('Used_CUs_per_timestep/ncus/', env.get_num_cus_pertimestep(association_action), time_step + 1)
            writer.add_scalar('Rate_Violation/viol/', info['rate_viol'], time_step + 1)
            writer.add_scalar('NRF_Violation/viol/', info.get('nrf_viol', 0), time_step + 1)

            # UPDATED per-timestep metrics
            writer.add_scalar('Spectral_Efficiency/timestep/', info['reward_components']['spectral_efficiency'], time_step + 1)
            writer.add_scalar('Sum_Rates/timestep/', info['reward_components']['sum_rates'], time_step + 1)
            writer.add_scalar('Deployment_Cost/timestep/', info['reward_components']['deployment_cost'], time_step + 1)
            writer.add_scalar('Main_Objective/timestep/', info['reward_components']['main_objective'], time_step + 1)
            writer.add_scalar('NRF_Lagrangian/timestep/', info['reward_components']['nrf_lagrangian'], time_step + 1)
            writer.add_scalar('Rate_Lagrangian/timestep/', info['reward_components']['rate_lagrangian'], time_step + 1)
            writer.add_scalar('Constraint_Penalties/timestep/nrf/', -np.sum(info['reward_components']['g2_constraints']), time_step + 1)
            writer.add_scalar('Constraint_Penalties/timestep/rate/', -np.sum(info['reward_components']['g3_constraints']), time_step + 1)
            writer.add_scalar('Rate_Violations/timestep/', info['rate_viol'], time_step + 1)
            writer.add_scalar('NRF_Violations/timestep/', info['nrf_viol'], time_step + 1)
            writer.add_scalar('Association_Violations/timestep/', info['reward_components']['association_violations'], time_step + 1)
            writer.add_scalar('QoS_Satisfaction/timestep/', info['qos_satisfaction'], time_step + 1)
            writer.add_scalar('Traffic_Demands/mean/', np.mean(info['traffic_demands']), time_step + 1)
            writer.add_scalar('Traffic_Demands/max/', np.max(info['traffic_demands']), time_step + 1)

            # SE-RL specific: Per-timestep projection metrics
            proj_info = info['projection_info']
            writer.add_scalar('Projections_Timestep/deployment_changed', int(proj_info['deployment_changed']), time_step + 1)
            writer.add_scalar('Projections_Timestep/association_changed', int(proj_info['association_changed']), time_step + 1)
            writer.add_scalar('Projections_Timestep/safety_deployments_added', proj_info['safety_deployments_added'], time_step + 1)
            writer.add_scalar('Projections_Timestep/validity_reassignments', proj_info['validity_reassignments'], time_step + 1)
            writer.add_scalar('Projections_Timestep/nrf_violations_fixed', proj_info['nrf_violations_fixed'], time_step + 1)
            writer.add_scalar('Projections_Timestep/dus_moved_for_nrf', proj_info['dus_moved_for_nrf'], time_step + 1)
            writer.add_scalar('Projections_Timestep/min_cus_required', proj_info['min_cus_required'], time_step + 1)
            
            # Individual projection penalties
            writer.add_scalar('Projection_Penalties_Timestep/total_penalty', info['reward_components']['projection_penalties'], time_step + 1)
            writer.add_scalar('Projection_Penalties_Timestep/safety_penalty', info['reward_components']['safety_penalty'], time_step + 1)
            writer.add_scalar('Projection_Penalties_Timestep/validity_penalty', info['reward_components']['validity_penalty'], time_step + 1)
            writer.add_scalar('Projection_Penalties_Timestep/nrf_violation_penalty', info['reward_components']['nrf_violation_penalty'], time_step + 1)
            writer.add_scalar('Projection_Penalties_Timestep/dus_moved_penalty', info['reward_components']['dus_moved_penalty'], time_step + 1)

            if PA>0:
                writer.add_scalar(f'Sumrate/tpt/train/{PA_NAME}/', np.sum(env.rate_list), time_step + 1)
                writer.add_scalar(f'Reward/reward/train/{PA_NAME}/', reward, time_step + 1)
                for d in range(D):
                    writer.add_scalar('DU_' + str(d + 1) + f'/association/train/{PA_NAME}/', association_action[d], time_step + 1)
                    writer.add_scalar('DU_' + str(d + 1) + f'/rate/train/{PA_NAME}/', env.rate_list[d], time_step + 1)
                    writer.add_scalar('DU_' + str(d + 1) + f'/demand/train/{PA_NAME}/', info['traffic_demands'][d], time_step + 1)

            print(f"Current timestep: {time_step}, episode: {i_episode}")
            time_step += 1
            
            # Update policy
            if time_step % update_timestep == 0:
                ppg_agent.update()
                env.update_demands()
                print("Lambda values before update : Lambda2 = "+ str(env.lambda_2)+ ": Lambda3 = " + str(env.lambda_3))
                env.lambda_2 = ppg_agent.lambda_2
                env.lambda_3 = ppg_agent.lambda_3
                print("Lambda values After update : Lambda2 = "+ str(env.lambda_2)+ ": Lambda3 = " + str(env.lambda_3))

            # Decay action standard deviation
            if time_step % action_std_decay_freq == 0:
                ppg_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # Decay temperature
            if time_step % decay_temp_timestep == 0:
                ppg_agent.decay_temp()

            # Save model
            if time_step % (save_model_freq) == 0:
                print("Saving model at timestep " + str(time_step))
                print("--------------------------------------------------------------------------------------------")
                agent_checkpoint_path = directory + "Model_ep_{}_ts_{}.pth".format(i_episode, time_step-1)
                ppg_agent.save(
                    agent_checkpoint_path, time_step, ppg_agent.optimizer,
                    ppg_agent.training_epoch, ppg_agent.lambda_rate,
                    ppg_agent.lambda_nrf, ppg_agent.mu_rate, ppg_agent.mu_nrf,
                    ppg_agent.beta_deploy, ppg_agent.beta_assoc, ppg_agent.entropy_weight
                )

            if t >= max_ep_len:
                break

        # ============= EPISODE COMPLETED - COMPREHENSIVE AGGREGATION =============
        
        episode_rewards.append(current_ep_reward)
        episode_lengths.append(current_ep_length)
        episode_rate_violations.append(np.mean(ep_rate_violations))
        episode_nrf_violations.append(np.mean(ep_nrf_violations))
        episode_qos_satisfactions.append(np.mean(ep_qos_satisfactions))
        episode_spectral_efficiencies.append(np.mean(ep_spectral_efficiencies))
        episode_sum_rates.append(np.mean(ep_sum_rates))                         
        episode_deployment_costs.append(np.mean(ep_deployment_costs))           
        episode_main_rewards.append(np.mean(ep_main_rewards))                   
        episode_association_violations.append(np.mean(ep_association_violations))

        episode_projection_stats.append(ep_projection_stats)
        episode_projection_penalties_list.append(episode_projection_penalties)

        # Update rolling averages
        window_size_actual = min(100, len(episode_rewards))
        avg_reward = np.mean(episode_rewards[-window_size_actual:])
        avg_ep_len = np.mean(episode_lengths[-window_size_actual:])
        avg_rate_viol = np.mean(episode_rate_violations[-window_size_actual:])
        avg_nrf_viol = np.mean(episode_nrf_violations[-window_size_actual:])
        avg_qos_satisfaction = np.mean(episode_qos_satisfactions[-window_size_actual:])
        avg_spectral_efficiency = np.mean(episode_spectral_efficiencies[-window_size_actual:])
        avg_sum_rates = np.mean(episode_sum_rates[-window_size_actual:])                         
        avg_deployment_cost = np.mean(episode_deployment_costs[-window_size_actual:])           
        avg_main_reward = np.mean(episode_main_rewards[-window_size_actual:])                   
        avg_association_viol = np.mean(episode_association_violations[-window_size_actual:])
        avg_projection_penalty = np.mean(episode_projection_penalties_list[-window_size_actual:])

        print_running_reward += current_ep_reward
        print_running_episodes += 1
        log_running_reward += current_ep_reward
        log_running_episodes += 1

        # Print progress
        if i_episode % print_freq == 0:
            running_time = (datetime.now().replace(microsecond=0) - start_time).total_seconds()
            print("============================================================================================")
            print("Episode: " + str(i_episode) + " | Timestep: " + str(time_step) + " | Runtime: " + str(running_time) + "s")
            print("Avg Reward: " + str(avg_reward) + " | QoS Satisfaction: " + str(avg_qos_satisfaction))
            print("Rate Violations: " + str(avg_rate_viol) + " | NRF Violations: " + str(avg_nrf_viol))
            print("Sum Rates: " + str(avg_sum_rates) + " | Deployment Cost: " + str(avg_deployment_cost))
            print("Main Reward: " + str(avg_main_reward) + " | Spectral Efficiency: " + str(avg_spectral_efficiency))
            print("============================================================================================")

        current_ep_sum_rate /= current_ep_length
        current_ep_main_obj /= current_ep_length
        current_ep_reward /= current_ep_length
        current_ep_qos_satisfaction /= current_ep_length

        avg_ep_rate_viol = np.sum(ep_rate_violations)/current_ep_length
        avg_ep_nrf_viol = np.sum(ep_nrf_violations)/current_ep_length

        # TensorBoard logging (per episode)
        writer.add_scalar('Reward/Average', avg_reward, i_episode)
        writer.add_scalar('Episode/Length', avg_ep_len, i_episode)
        writer.add_scalar('Violations/Rate', avg_ep_rate_viol, i_episode)
        writer.add_scalar('Violations/NRF', avg_ep_nrf_viol, i_episode)
        writer.add_scalar('Performance/QoS_Satisfaction', avg_qos_satisfaction, i_episode)
        writer.add_scalar('Performance/Spectral_Efficiency', avg_spectral_efficiency, i_episode)
        writer.add_scalar('Performance/Sum_Rates', avg_sum_rates, i_episode)                         
        writer.add_scalar('Performance/Deployment_Cost', avg_deployment_cost, i_episode)             
        writer.add_scalar('Reward/Main_Average', avg_main_reward, i_episode)                         
        writer.add_scalar('Violations/Association', avg_association_viol, i_episode)
        writer.add_scalar('Hyperparameters/Temperature', ppg_agent.temp, i_episode)
        writer.add_scalar('Hyperparameters/Lambda_Rate', ppg_agent.lambda_rate, i_episode)
        writer.add_scalar('Hyperparameters/Lambda_NRF', ppg_agent.lambda_nrf, i_episode)
        writer.add_scalar('Hyperparameters/Mu_Rate', ppg_agent.mu_rate, i_episode)
        writer.add_scalar('Hyperparameters/Mu_NRF', ppg_agent.mu_nrf, i_episode)
        writer.add_scalar('Hyperparameters/Beta_Deploy', ppg_agent.beta_deploy, i_episode)
        writer.add_scalar('Hyperparameters/Beta_Assoc', ppg_agent.beta_assoc, i_episode)
        writer.add_scalar('Hyperparameters/Entropy_Weight', ppg_agent.entropy_weight, i_episode)
        writer.add_scalar('Hyperparameters/Action_Std', ppg_agent.policy.action_std, i_episode)
        writer.add_scalar(f'Sumrate/tpt/train/{PA_NAME}/ep', current_ep_sum_rate, i_episode)
        writer.add_scalar(f'Reward/reward/train/{PA_NAME}/ep', current_ep_reward, i_episode)
        writer.add_scalar('Main_Objective/ep/', current_ep_main_obj, i_episode)                   
        writer.add_scalar('Rate_Violations/ep/', avg_ep_rate_viol, i_episode)
        writer.add_scalar('NRF_Violations/ep/', avg_ep_nrf_viol, i_episode)
        writer.add_scalar('QoS_Satisfaction/ep/', current_ep_qos_satisfaction, time_step + 1)

        # SE-RL: Original episode-level projection metrics
        ep_proj_deployments_changed = np.mean([p['deployment_changed'] for p in ep_projection_stats])
        ep_proj_associations_changed = np.mean([p['association_changed'] for p in ep_projection_stats])
        ep_proj_safety_added = np.mean([p['safety_deployments_added'] for p in ep_projection_stats])
        ep_proj_validity_reassign = np.mean([p['validity_reassignments'] for p in ep_projection_stats])
        ep_proj_nrf_fixed = np.mean([p['nrf_violations_fixed'] for p in ep_projection_stats])
        ep_proj_dus_moved = np.mean([p['dus_moved_for_nrf'] for p in ep_projection_stats])
        ep_proj_min_cus_req = np.mean([p['min_cus_required'] for p in ep_projection_stats])
        
        writer.add_scalar('Projections_Episode/deployment_changed_rate', ep_proj_deployments_changed, i_episode)
        writer.add_scalar('Projections_Episode/association_changed_rate', ep_proj_associations_changed, i_episode)
        writer.add_scalar('Projections_Episode/avg_safety_deployments_added', ep_proj_safety_added, i_episode)
        writer.add_scalar('Projections_Episode/avg_validity_reassignments', ep_proj_validity_reassign, i_episode)
        writer.add_scalar('Projections_Episode/avg_nrf_violations_fixed', ep_proj_nrf_fixed, i_episode)
        writer.add_scalar('Projections_Episode/avg_dus_moved_for_nrf', ep_proj_dus_moved, i_episode)
        writer.add_scalar('Projections_Episode/min_cus_required', ep_proj_min_cus_req, i_episode)
        writer.add_scalar('Projection_Penalties_Episode/total_penalty', episode_projection_penalties, i_episode)
        writer.add_scalar('Projection_Penalties_Episode/avg_penalty', avg_projection_penalty, i_episode)

        # ============= NEW: COMPREHENSIVE EPISODE-LEVEL DIAGNOSTICS =============
        
        # 1. Deployment tracking (Raw vs Projected CUs)
        writer.add_scalar('Deployment_Episode/avg_raw_cus', np.mean(ep_raw_cus), i_episode)
        writer.add_scalar('Deployment_Episode/avg_projected_cus', np.mean(ep_projected_cus), i_episode)
        writer.add_scalar('Deployment_Episode/avg_cus_gap', np.mean(ep_cus_gap), i_episode)
        
        # 2. Diagnostic ratios
        writer.add_scalar('Diagnostics_Episode/avg_penalty_objective_ratio', np.mean(ep_penalty_obj_ratios), i_episode)
        writer.add_scalar('Exploration_Episode/avg_deployment_diversity', np.mean(ep_deployment_diversity), i_episode)
        
        # 3. Success rates (% of timesteps with 0 violations - KEY METRIC!)
        safety_success_rate = np.mean(ep_safety_success)
        validity_success_rate = np.mean(ep_validity_success)
        nrf_success_rate = np.mean(ep_nrf_success)
        overall_success_rate = safety_success_rate * validity_success_rate * nrf_success_rate
        
        writer.add_scalar('Success_Rates/safety_success_rate', safety_success_rate, i_episode)
        writer.add_scalar('Success_Rates/validity_success_rate', validity_success_rate, i_episode)
        writer.add_scalar('Success_Rates/nrf_success_rate', nrf_success_rate, i_episode)
        writer.add_scalar('Success_Rates/overall_success_rate', overall_success_rate, i_episode)
        
        # 4. Penalty breakdown (which constraint is most problematic)
        total_ep_penalty = abs(episode_projection_penalties)
        if total_ep_penalty > 1e-6:
            ep_infos = ppg_agent.buffer.infos
            safety_total = abs(sum([info['reward_components']['safety_penalty'] for info in ep_infos]))
            validity_total = abs(sum([info['reward_components']['validity_penalty'] for info in ep_infos]))
            nrf_total = abs(sum([info['reward_components']['nrf_violation_penalty'] for info in ep_infos]))
            movement_total = abs(sum([info['reward_components']['dus_moved_penalty'] for info in ep_infos]))
            
            writer.add_scalar('Penalty_Breakdown/safety_fraction', safety_total / total_ep_penalty, i_episode)
            writer.add_scalar('Penalty_Breakdown/validity_fraction', validity_total / total_ep_penalty, i_episode)
            writer.add_scalar('Penalty_Breakdown/nrf_fraction', nrf_total / total_ep_penalty, i_episode)
            writer.add_scalar('Penalty_Breakdown/movement_fraction', movement_total / total_ep_penalty, i_episode)
        
        # 5. CONVERGENCE INDICATOR - THE MOST IMPORTANT SINGLE METRIC!
        # Score from 0 to 1, higher = better learning
        convergence_score = (
            (1.0 - min(1.0, ep_proj_safety_added / max(1, env.min_cus_required))) *
            (1.0 - min(1.0, ep_proj_validity_reassign / max(1, D))) *
            (1.0 - min(1.0, ep_proj_nrf_fixed / max(1, C))) *
            avg_qos_satisfaction
        )
        writer.add_scalar('Convergence/episode_score', convergence_score, i_episode)
        
        # Rolling convergence (smoothed version - WATCH THIS!)
        convergence_history.append(convergence_score)
        rolling_convergence = np.mean(convergence_history[-min(window_size, len(convergence_history)):])
        writer.add_scalar('Convergence/rolling_average', rolling_convergence, i_episode)

        # Optional post-convergence learning-rate ablation. The default factor
        # 1.0 preserves the faithful/T6 behavior exactly. T9 changes only this
        # factor, after episode 200, to test whether continued large PPO steps
        # cause the observed loss of an already-converged policy.
        post200_lr_factor = float(os.environ.get("CHARM_POST200_LR_FACTOR", "1.0"))
        if i_episode == 200 and post200_lr_factor != 1.0:
            for param_group in ppg_agent.optimizer.param_groups:
                param_group['lr'] *= post200_lr_factor
            writer.add_scalar(
                'Learning_Rate/post200_factor', post200_lr_factor, i_episode
            )
            writer.add_scalar(
                'Learning_Rate/actor_post200',
                ppg_agent.optimizer.param_groups[0]['lr'], i_episode
            )
            writer.add_scalar(
                'Learning_Rate/critic_post200',
                ppg_agent.optimizer.param_groups[1]['lr'], i_episode
            )
            print(
                f"POST-200 LR FACTOR applied: {post200_lr_factor}; "
                f"actor={ppg_agent.optimizer.param_groups[0]['lr']}, "
                f"critic={ppg_agent.optimizer.param_groups[1]['lr']}"
            )
        
        # ============= FIXED ADAPTIVE PENALTIES =============
        penalty_schedule = os.environ.get("CHARM_PENALTY_SCHEDULE", "conditional")
        if penalty_schedule == "dec8_uniform" and i_episode % 50 == 0:
            # Replay the trajectory recorded in the Dec-8 TensorBoard export:
            # all four projection multipliers increase by 10% at episodes
            # 0, 50, ..., 800. This is the sole algorithmic change in TUNE-1.
            env.penalty_multipliers['safety'] *= 1.1
            env.penalty_multipliers['nrf_violation'] *= 1.1
            env.penalty_multipliers['validity'] *= 1.1
            env.penalty_multipliers['movement'] *= 1.1
            writer.add_scalar('Adaptive_Penalties/safety_multiplier', env.penalty_multipliers['safety'], i_episode)
            writer.add_scalar('Adaptive_Penalties/nrf_multiplier', env.penalty_multipliers['nrf_violation'], i_episode)
            writer.add_scalar('Adaptive_Penalties/validity_multiplier', env.penalty_multipliers['validity'], i_episode)
            writer.add_scalar('Adaptive_Penalties/movement_multiplier', env.penalty_multipliers['movement'], i_episode)
            print(f"DEC8 UNIFORM PENALTY REPLAY at episode {i_episode}: {env.penalty_multipliers}")

        # ONLY runs episodes 0-200, NO penalty reduction
        elif i_episode % 50 == 0:  #and i_episode > 0 and i_episode <= 200:    #if i_episode % 20 == 0:  #and i_episode > 0 and i_episode <= 200:
            # Calculate recent averages
            recent_penalty_ratios = ep_penalty_obj_ratios[-20:] if len(ep_penalty_obj_ratios) >= 20 else ep_penalty_obj_ratios
            avg_penalty_ratio = np.mean(recent_penalty_ratios)
            
            recent_episodes = min(20, len(episode_projection_stats))
            recent_safety_violations = [np.mean([p['safety_deployments_added'] for p in ep]) 
                                       for ep in episode_projection_stats[-recent_episodes:]]
            recent_nrf_violations = [np.mean([p['nrf_violations_fixed'] for p in ep]) 
                                    for ep in episode_projection_stats[-recent_episodes:]]
            recent_validity_violations = [np.mean([p['validity_reassignments'] for p in ep]) 
                                         for ep in episode_projection_stats[-recent_episodes:]]
            
            avg_safety_violations = np.mean(recent_safety_violations)
            avg_nrf_violations = np.mean(recent_nrf_violations)
            avg_validity_violations = np.mean(recent_validity_violations)
            
            writer.add_scalar('Adaptive_Penalties/penalty_objective_ratio', avg_penalty_ratio, i_episode)
            writer.add_scalar('Adaptive_Penalties/avg_safety_violations', avg_safety_violations, i_episode)
            writer.add_scalar('Adaptive_Penalties/avg_nrf_violations', avg_nrf_violations, i_episode)
            writer.add_scalar('Adaptive_Penalties/avg_validity_violations', avg_validity_violations, i_episode)
            
            print(f"\n{'='*80}")
            print(f"FIXED ADAPTIVE PENALTY CHECK (Episode {i_episode})")
            print(f"{'='*80}")
            print(f"  Current Penalty Multipliers:")
            print(f"    Safety:   {env.penalty_multipliers['safety']:.2f}")
            print(f"    NRF:      {env.penalty_multipliers['nrf_violation']:.2f}")
            print(f"    Validity: {env.penalty_multipliers['validity']:.2f}")
            print(f"  Avg Penalty/Objective Ratio: {avg_penalty_ratio:.3f}")
            print(f"  Success Rates: Safety={safety_success_rate:.2%}, NRF={nrf_success_rate:.2%}, Overall={overall_success_rate:.2%}")
            
            adjustment_made = False
            
            # ONLY INCREASE penalties when needed (NO reduction logic)
            if avg_penalty_ratio < 0.7 and overall_success_rate < 0.9:
                env.penalty_multipliers['safety'] *= 1.1
                env.penalty_multipliers['nrf_violation'] *= 1.1
                env.penalty_multipliers['validity'] *= 1.1
                env.penalty_multipliers['movement'] *= 1.1
                print(f"  ⚠️  ACTION: Penalties too weak - Increasing by 10%")
                adjustment_made = True
                
            # Targeted increases for specific constraints
            elif not adjustment_made:
                if avg_safety_violations > 0.5 and safety_success_rate < 0.9:
                    env.penalty_multipliers['safety'] *= 1.2
                    print(f"  ⚠️  ACTION: Safety violations high - Increasing safety penalty by 32%")
                    adjustment_made = True
                    
                if avg_nrf_violations > 0.5 and nrf_success_rate < 0.9:
                    env.penalty_multipliers['nrf_violation'] *= 1.2
                    env.penalty_multipliers['movement'] *= 1.2
                    print(f"  ⚠️  ACTION: NRF violations high - Increasing NRF penalties")
                    adjustment_made = True
                    
                if avg_validity_violations > 0.5 and validity_success_rate < 0.9:   # changed from 0.7 to 0.9
                    env.penalty_multipliers['validity'] *= 1.2
                    print(f"  ⚠️  ACTION: Validity violations high - Increasing validity penalty by  20%")
                    adjustment_made = True
            
            if not adjustment_made:
                print(f"  ✓ Penalties adequate - no adjustment needed")
            
            if adjustment_made:
                print(f"  New Penalty Multipliers:")
                print(f"    Safety:   {env.penalty_multipliers['safety']:.2f}")
                print(f"    NRF:      {env.penalty_multipliers['nrf_violation']:.2f}")
                print(f"    Validity: {env.penalty_multipliers['validity']:.2f}")
                
                writer.add_scalar('Adaptive_Penalties/safety_multiplier', env.penalty_multipliers['safety'], i_episode)
                writer.add_scalar('Adaptive_Penalties/nrf_multiplier', env.penalty_multipliers['nrf_violation'], i_episode)
                writer.add_scalar('Adaptive_Penalties/validity_multiplier', env.penalty_multipliers['validity'], i_episode)
            
            print(f"{'='*80}\n")
        
        elif i_episode == 200:
            print(f"\n{'='*80}")
            print(f"PENALTIES FROZEN at Episode 200")
            print(f"  Final Penalty Multipliers:")
            print(f"    Safety:   {env.penalty_multipliers['safety']:.2f}")
            print(f"    NRF:      {env.penalty_multipliers['nrf_violation']:.2f}")
            print(f"    Validity: {env.penalty_multipliers['validity']:.2f}")
            print(f"  These will remain constant for rest of training")
            print(f"{'='*80}\n")
        # ============= END FIXED ADAPTIVE PENALTIES =============
        
        # Alert if convergence stalling
        if i_episode > 0 and i_episode % 100 == 0:
            print(f"\n*** CONVERGENCE CHECK (Episode {i_episode}) ***")
            print(f"    Rolling Convergence Score: {rolling_convergence:.4f}")
            print(f"    Safety Success Rate: {safety_success_rate:.2%}")
            print(f"    Validity Success Rate: {validity_success_rate:.2%}")
            print(f"    NRF Success Rate: {nrf_success_rate:.2%}")
            print(f"    Overall Success Rate: {overall_success_rate:.2%}")
            print(f"    Avg Raw CUs: {np.mean(ep_raw_cus):.2f} (target: {env.min_cus_required})")
            print(f"    Penalty/Obj Ratio: {np.mean(ep_penalty_obj_ratios):.3f}")
            if rolling_convergence < 0.5 and i_episode >= 300:
                print(f"    ⚠️  WARNING: Convergence score below 0.5 at episode {i_episode}")
                print(f"        Consider adjusting penalties or increasing exploration!")
            elif rolling_convergence >= 0.8:
                print(f"    ✅ EXCELLENT: Policy is mastering constraints!")
            print("*" * 50 + "\n")
        
        # ==================================================================

        if i_episode % log_freq == 0:
            # Store training data
            training_data.append({
                'episode': i_episode,
                'timestep': time_step,
                'avg_reward': avg_reward,
                'avg_episode_length': avg_ep_len,
                'avg_rate_violation': avg_rate_viol,
                'avg_nrf_violation': avg_nrf_viol,
                'avg_qos_satisfaction': avg_qos_satisfaction,
                'avg_spectral_efficiency': avg_spectral_efficiency,
                'avg_sum_rates': avg_sum_rates,                         
                'avg_deployment_cost': avg_deployment_cost,             
                'avg_main_reward': avg_main_reward,                     
                'avg_association_violation': avg_association_viol,
                'temperature': ppg_agent.temp,
                'lambda_rate': ppg_agent.lambda_rate,
                'lambda_nrf': ppg_agent.lambda_nrf,
                'mu_rate': ppg_agent.mu_rate,
                'mu_nrf': ppg_agent.mu_nrf,
                'beta_deploy': ppg_agent.beta_deploy,
                'beta_assoc': ppg_agent.beta_assoc,
                'entropy_weight': ppg_agent.entropy_weight,
                'action_std': ppg_agent.policy.action_std,
                'episode_projection_penalties': episode_projection_penalties,
                'avg_projection_penalty': avg_projection_penalty,
                'ep_proj_deployments_changed': ep_proj_deployments_changed,
                'ep_proj_associations_changed': ep_proj_associations_changed,
                'ep_proj_safety_added': ep_proj_safety_added,
                'ep_proj_validity_reassign': ep_proj_validity_reassign,
                'ep_proj_nrf_fixed': ep_proj_nrf_fixed,
                'ep_proj_dus_moved': ep_proj_dus_moved,
                'ep_proj_min_cus_req': ep_proj_min_cus_req,
                # NEW diagnostic metrics
                'avg_raw_cus': np.mean(ep_raw_cus),
                'avg_projected_cus': np.mean(ep_projected_cus),
                'avg_cus_gap': np.mean(ep_cus_gap),
                'avg_penalty_obj_ratio': np.mean(ep_penalty_obj_ratios),
                'safety_success_rate': safety_success_rate,
                'validity_success_rate': validity_success_rate,
                'nrf_success_rate': nrf_success_rate,
                'overall_success_rate': overall_success_rate,
                'convergence_score': convergence_score,
                'rolling_convergence': rolling_convergence,
                # Adaptive penalties
                'penalty_safety': env.penalty_multipliers['safety'],
                'penalty_nrf': env.penalty_multipliers['nrf_violation'],
                'penalty_validity': env.penalty_multipliers['validity'],
                'penalty_movement': env.penalty_multipliers['movement']
            })

        if time_step % 500 == 0:
            ppg_agent.scheduler.step()
            print("Learning rates: actor=" + str(ppg_agent.optimizer.param_groups[0]['lr']) + ", critic=" + str(ppg_agent.optimizer.param_groups[1]['lr']))

        # Full-state checkpoints are written only after all episode-level
        # updates. Resuming them starts at the next reset with identical RNG,
        # rollout buffers, optimizer/scheduler, penalties, and histories.
        completed_episodes = i_episode + 1
        full_checkpoint_due = (
            completed_episodes % full_checkpoint_freq == 0
            or time_step >= max_training_timesteps
        )
        if full_checkpoint_due:
            full_checkpoint_path = os.path.join(
                directory,
                f"FullState_ep_{completed_episodes:03d}_ts_{time_step}.pth",
            )
            eligible_best = rolling_convergence >= 0.95
            improved_best = (
                eligible_best
                and avg_main_reward > best_state["rolling_main_objective"] + 0.05
            )
            if improved_best:
                best_state = {
                    "checkpoint": full_checkpoint_path,
                    "completed_episodes": completed_episodes,
                    "timestep": time_step,
                    "rolling_convergence": float(rolling_convergence),
                    "rolling_main_objective": float(avg_main_reward),
                }

            histories = {
                "training_data": training_data,
                "episode_rewards": episode_rewards,
                "episode_lengths": episode_lengths,
                "episode_rate_violations": episode_rate_violations,
                "episode_nrf_violations": episode_nrf_violations,
                "episode_qos_satisfactions": episode_qos_satisfactions,
                "episode_spectral_efficiencies": episode_spectral_efficiencies,
                "episode_traffic_utilities": episode_traffic_utilities,
                "episode_sum_rates": episode_sum_rates,
                "episode_deployment_costs": episode_deployment_costs,
                "episode_main_rewards": episode_main_rewards,
                "episode_secondary_rewards": episode_secondary_rewards,
                "episode_constraint_penalties": episode_constraint_penalties,
                "episode_association_violations": episode_association_violations,
                "episode_load_balances": episode_load_balances,
                "episode_projection_stats": episode_projection_stats,
                "episode_projection_penalties_list": episode_projection_penalties_list,
                "convergence_history": convergence_history,
                "CU_usage": CU_usage,
            }
            save_full_checkpoint(
                full_checkpoint_path,
                agent=ppg_agent,
                env=env,
                next_episode=completed_episodes,
                timestep=time_step,
                histories=histories,
                best_state=best_state,
                run_config=checkpoint_run_config,
            )
            if improved_best:
                manifest_path = os.path.join(directory, "best_checkpoint.json")
                manifest_tmp = manifest_path + ".tmp"
                with open(manifest_tmp, "w", encoding="utf-8") as manifest_file:
                    json.dump(best_state, manifest_file, indent=2)
                os.replace(manifest_tmp, manifest_path)
                print(f"New best full-state checkpoint: {full_checkpoint_path}")
        
        end_time_eps = datetime.now().replace(microsecond=0)
        print(f"Time taken for episode: {end_time_eps - start_time_eps}")
        i_episode += 1
        
        if time_step >= max_training_timesteps:
            print('Check if it still not converged and increase episodes or max steps')
    
    # Training completed
    print("============================================================================================")
    print("SE-RL TRAINING WITH ADAPTIVE PENALTIES COMPLETED!")
    end_time = datetime.now().replace(microsecond=0)
    total_time = end_time - start_time
    print("Total training time: " + str(total_time))
    print("Episodes: " + str(i_episode) + " | Timesteps: " + str(time_step))
    print("============================================================================================")

    # Save final model
    print("Saving model at timestep " + str(time_step))
    print("--------------------------------------------------------------------------------------------")
    final_checkpoint_path = directory + "Model_ep_{}_ts_{}_final.pth".format(i_episode, time_step)
    try:
        ppg_agent.save(
            final_checkpoint_path, time_step, ppg_agent.optimizer,
            ppg_agent.training_epoch, ppg_agent.lambda_rate, ppg_agent.lambda_nrf,
            ppg_agent.mu_rate, ppg_agent.mu_nrf, ppg_agent.beta_deploy,
            ppg_agent.beta_assoc, ppg_agent.entropy_weight
        )
    except (OSError, RuntimeError) as checkpoint_error:
        # Artifact I/O must never discard an otherwise completed trajectory.
        print("WARNING: final checkpoint save failed; continuing to save logs: "
              + str(checkpoint_error))

    # Save training logs
    df = pd.DataFrame(training_data)
    df.to_csv(log_f_name, index=False)
    
    with open(train_experiences_log_f_name, 'wb') as f:
        pickle.dump({
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'episode_rate_violations': episode_rate_violations,
            'episode_nrf_violations': episode_nrf_violations,
            'episode_qos_satisfactions': episode_qos_satisfactions,
            'episode_spectral_efficiencies': episode_spectral_efficiencies,
            'episode_sum_rates': episode_sum_rates,                         
            'episode_deployment_costs': episode_deployment_costs,           
            'episode_main_rewards': episode_main_rewards,                   
            'episode_association_violations': episode_association_violations,
            'episode_projection_stats': episode_projection_stats,           
            'episode_projection_penalties_list': episode_projection_penalties_list,
            'convergence_history': convergence_history,
            'training_data': training_data
        }, f)

    # Final statistics
    print("============================================================================================")
    print("FINAL SE-RL PERFORMANCE STATISTICS")
    print("Average Reward (last " + str(window_size) + " episodes): " + str(avg_reward))
    print("Average Episode Length: " + str(avg_ep_len))
    print("Rate Violation Rate: " + str(avg_rate_viol))
    print("NRF Violation Rate: " + str(avg_nrf_viol))
    print("QoS Satisfaction Rate: " + str(avg_qos_satisfaction))
    print("Spectral Efficiency: " + str(avg_spectral_efficiency))
    print("Sum Rates: " + str(avg_sum_rates))
    print("Deployment Cost: " + str(avg_deployment_cost))
    print("Main Reward: " + str(avg_main_reward))
    print("Association Violations: " + str(avg_association_viol))
    print("SE-RL Average Projection Penalty: " + str(avg_projection_penalty))
    print(f"FINAL CONVERGENCE SCORE: {rolling_convergence:.4f}")
    print(f"FINAL PENALTY MULTIPLIERS:")
    print(f"  Safety:   {env.penalty_multipliers['safety']:.2f}")
    print(f"  NRF:      {env.penalty_multipliers['nrf_violation']:.2f}")
    print(f"  Validity: {env.penalty_multipliers['validity']:.2f}")
    print(f"  Movement: {env.penalty_multipliers['movement']:.2f}")
    if rolling_convergence >= 0.8:
        print("✅ EXCELLENT CONVERGENCE - Policy mastered constraints!")
    elif rolling_convergence >= 0.6:
        print("⚠️  MODERATE CONVERGENCE - Policy learned but could improve")
    else:
        print("❌ POOR CONVERGENCE - Consider retraining with adjusted parameters")
    print("============================================================================================")

    writer.close()
    env.close()

    return ppg_agent, final_checkpoint_path

if __name__ == '__main__':
    try:
        agent, checkpoint_path = train()
        print("SE-RL Training with adaptive penalties completed! Model saved at: " + checkpoint_path)
    except KeyboardInterrupt:
        print("SE-RL Training stopped by user")
    except Exception as e:
        print("SE-RL Training failed: " + str(e))
        traceback.print_exc()
