import os
import numpy as np
try:
    from gym import Env, spaces
except ImportError:
    from gymnasium import Env, spaces
import torch
from typing import Tuple
import math

class THzMidhaulEnv_SE_RL(Env):
    def __init__(self, C: int = 11, D: int = 43, scn=None, PA: int = 4, NRF: int = 12, lambda_sparsity: float = 10.0):
        super().__init__()
        self.C = C
        self.D = D
        self.NRF = NRF
        self.BW = 2e9
        self.min_p_dbm = -20
        self.max_p_dbm = 23
        self.min_p = 0
        self.max_p = self.dbmtolinear(self.max_p_dbm)
        self.min_power = 0.06 * self.max_p
        self.min_sinr_db = 0.1
        self.max_sinr_db = 10
        self.min_sinr = 0.01
        self.max_sinr = 3
        self.scn = scn
        self.PA = PA
        self.lambda_sparsity = lambda_sparsity
        
        # Calculate minimum CUs required generically
        self.min_cus_required = math.ceil(self.D / self.NRF)
        print(f"Generic minimum CUs required: {self.min_cus_required} (D={self.D}, NRF={self.NRF})")
        
        # FIXED: No curriculum - fixed minimum rate
        self.curr_min_rate = 6.0  # Fixed R_min = 6.0 Gbps
        
        # Fixed Lagrangian multipliers
        self.lambda_2 = 10.0   # NRF multiplier
        self.lambda_3 = 10.0  # Rate multiplier  

        # Traffic model
        self.timestep = 0
        self.episode_count = 0
        self.max_steps = 50  # For truncation
        
        # State tracking for trends
        self.prev_rates = np.zeros(D)
        self.prev_demands = np.zeros(D)
        self.current_demands = np.zeros(D)

        self.rate_list = []

        # Network-aware penalty scaling
        self.penalty_multipliers = self.get_network_scaled_penalties()
        print(f"Network-scaled penalty multipliers: {self.penalty_multipliers}")
        
        # Enhanced state space: D DUs × 6 features
        self.action_space = spaces.Tuple((
            spaces.MultiBinary(self.C), 
            spaces.MultiDiscrete([self.C] * self.D)
        ))
        self.observation_space = spaces.Box(
            self.min_sinr, self.max_sinr, 
            shape=(self.D * 6,), dtype=np.float32
        )

    def dbmtolinear(self, value: float) -> float:
        """Convert dBm to linear scale"""
        return 10 ** ((value - 30) / 10.0)
    
    def get_network_scaled_penalties(self):
        """
        Calculate penalty multipliers scaled for network size
        
        Analysis shows that penalty multipliers optimized for large networks (C=11, D=43) 
        are 2-3x too strong for smaller networks, disrupting learning dynamics.
        
        Solution: Scale penalties based on network complexity relative to base network.
        Target penalty/objective ratio: 0.5-1.0 for optimal learning.
        
        Returns:
            dict: Scaled penalty multipliers for current network configuration
        """
        # Base multipliers (optimized for C=11, D=43 base network)
        # Environment override supports controlled one-parameter tuning while
        # retaining the faithful December value (50) by default.
        BASE_SAFETY = float(os.environ.get("CHARM_BASE_SAFETY", "50.0"))
        BASE_NRF = float(os.environ.get("CHARM_BASE_NRF", "40.0"))
        BASE_VALIDITY = float(os.environ.get("CHARM_BASE_VALIDITY", "20.0"))
        BASE_MOVEMENT = float(os.environ.get("CHARM_BASE_MOVEMENT", "10.0"))
        
        # Calculate scaling factor based on network complexity
        base_network_size = 11 * 43  # 473 (base network for penalty optimization)
        current_network_size = self.C * self.D
        network_scale = current_network_size / base_network_size
        
        # Clamp scale factor between 20%-100% to prevent penalties from becoming too weak/strong
        scale_factor = max(0.2, min(1.0, network_scale))
        
        # Calculate expected penalty/objective ratio for validation
        expected_penalty_obj_ratio = 0.77 * scale_factor / network_scale if network_scale > 0 else 0.77
        
        scaled_penalties = {
            'safety': BASE_SAFETY * scale_factor,
            'nrf_violation': BASE_NRF * scale_factor,
            'validity': BASE_VALIDITY * scale_factor,
            'movement': BASE_MOVEMENT * scale_factor,
            # Metadata for analysis
            'scale_factor': scale_factor,
            'network_scale': network_scale,
            'expected_penalty_obj_ratio': expected_penalty_obj_ratio,
            'base_network_size': base_network_size,
            'current_network_size': current_network_size
        }
        
        print(f"Network Scaling Analysis:")
        print(f"  Base network: C=11, D=43 (size={base_network_size})")
        print(f"  Current network: C={self.C}, D={self.D} (size={current_network_size})")
        print(f"  Network scale: {network_scale:.3f}")
        print(f"  Applied scale factor: {scale_factor:.3f}")
        print(f"  Expected penalty/objective ratio: {expected_penalty_obj_ratio:.3f}")
        print(f"  Scaled penalties: Safety={scaled_penalties['safety']:.1f}, "
              f"NRF={scaled_penalties['nrf_violation']:.1f}, "
              f"Validity={scaled_penalties['validity']:.1f}, "
              f"Movement={scaled_penalties['movement']:.1f}")
        
        return scaled_penalties
    
    def update_demands(self):
        self.current_demands = np.random.uniform(self.curr_min_rate - 2.0, self.curr_min_rate + 2.0, self.D)

    def project_deployment_safety(self, x_i_raw: np.ndarray) -> np.ndarray:
        """
        SE-RL: Apply generic safety projection to ensure minimum required CUs are deployed
        Minimum CUs = ceil(D/NRF) to ensure all DUs can potentially be served
        """
        x_i = x_i_raw.copy()
        current_deployed = np.sum(x_i)
        
        # If insufficient CUs are deployed, deploy additional ones
        if current_deployed < self.min_cus_required:
            cus_to_deploy = self.min_cus_required - current_deployed
            
            # Find undeployed CUs and deploy the first 'cus_to_deploy' of them
            undeployed_cus = np.where(x_i == 0)[0]
            
            if len(undeployed_cus) >= cus_to_deploy:
                # Deploy the first required number of undeployed CUs
                for i in range(cus_to_deploy):
                    x_i[undeployed_cus[i]] = 1
            else:
                # Deploy all available undeployed CUs (edge case)
                x_i[undeployed_cus] = 1
                print(f"WARNING: Only {len(undeployed_cus)} CUs available, need {cus_to_deploy} more")
            
        return x_i

    def project_association_validity(self, association_action_raw: np.ndarray, x_i: np.ndarray) -> np.ndarray:
        """
        SE-RL: Apply association validity projection to ensure only deployed CUs are associated
        Note: This is called AFTER safety projection, so deployed_cus is guaranteed to be non-empty
        """
        association_action = association_action_raw.copy()
        deployed_cus = np.where(x_i == 1)[0]
        
        # For each DU, if it's associated to an undeployed CU, reassign to first deployed CU
        for du_idx in range(self.D):
            if association_action[du_idx] not in deployed_cus:
                association_action[du_idx] = deployed_cus[0]
                
        return association_action

    def project_nrf_constraints(self, association_action: np.ndarray, x_i: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        SE-RL: Apply NRF projection to ensure no CU exceeds NRF capacity
        """
        association_action_projected = association_action.copy()
        cu_loads = np.bincount(association_action_projected, minlength=self.C)
        deployed_cus = np.where(x_i == 1)[0]
        
        nrf_violations_fixed = 0
        dus_moved_for_nrf = 0
        
        # Handle NRF violations
        for cu in deployed_cus:
            current_load = cu_loads[cu]
            if current_load <= self.NRF:
                continue
                
            # This CU is overloaded
            nrf_violations_fixed += 1
            excess = current_load - self.NRF
            
            # Find DUs currently assigned to this overloaded CU
            dus_assigned_to_cu = np.where(association_action_projected == cu)[0]
            
            # Move the last 'excess' DUs to other deployed CUs with capacity
            dus_to_move = dus_assigned_to_cu[-excess:]
            
            for du_idx in dus_to_move:
                # Find deployed CUs with available capacity
                available_cus = []
                for candidate_cu in deployed_cus:
                    if cu_loads[candidate_cu] < self.NRF:
                        available_cus.append(candidate_cu)
                
                if available_cus:
                    # Move to the first available CU
                    target_cu = available_cus[0]
                    association_action_projected[du_idx] = target_cu
                    cu_loads[cu] -= 1
                    cu_loads[target_cu] += 1
                    dus_moved_for_nrf += 1
                else:
                    # No capacity available - this shouldn't happen with proper min_cus_required
                    print(f"WARNING: No capacity available for DU {du_idx}")
        
        final_cu_loads = np.bincount(association_action_projected, minlength=self.C)
        
        nrf_info = {
            'nrf_violations_fixed': nrf_violations_fixed,
            'dus_moved_for_nrf': dus_moved_for_nrf,
            'final_cu_loads': final_cu_loads
        }
        
        return association_action_projected, nrf_info

    def reset(self, seed=None, options=None) -> np.ndarray:
        """Reset environment to initial state"""
        if seed is not None:
            np.random.seed(seed)
        
        self.timestep = 0
        self.episode_count += 1
        
        # Update demands for new episode
        #self.update_demands()
        
        # Generate initial conditions
        initial_rates = np.random.uniform(1.0, 5.0, self.D)
        
        # Create enhanced state [D*6]
        enhanced_state = self.create_enhanced_state(initial_rates, self.current_demands)
        
        # Update tracking
        self.prev_rates = initial_rates.copy()
        self.prev_demands = self.current_demands.copy()
        self.rate_list = np.zeros(self.D)
        
        return enhanced_state

    def create_enhanced_state(self, rates: np.ndarray, demands: np.ndarray) -> np.ndarray:
        """
        Create enhanced state [D*6] = D DUs × 6 features
        """
        # Normalize rates and demands
        normalized_rates = 3.0 * np.tanh(rates / 8.0)
        normalized_demands = 3.0 * np.tanh(demands / 6.0)
        
        # QoS satisfaction ratio
        qos_ratio = np.tanh(rates / (demands + 1e-8))
        
        # Demand urgency (higher for underserved)
        urgency = np.tanh((demands - rates) / (demands + 1e-8))
        urgency = np.maximum(0, urgency)
        
        # Trends
        rate_trend = np.tanh((rates - self.prev_rates) / 2.0)
        demand_trend = np.tanh((demands - self.prev_demands) / 2.0)
        
        # Interleave features
        enhanced_state = np.empty(self.D * 6, dtype=np.float32)
        for i in range(self.D):
            base_idx = i * 6
            enhanced_state[base_idx:base_idx+6] = [
                normalized_rates[i], normalized_demands[i], qos_ratio[i],
                urgency[i], rate_trend[i], demand_trend[i]
            ]
        
        return enhanced_state

    def step(self, action: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one step in the environment"""
        # Update demands for dynamic traffic
        #self.update_demands()
        
        x_i_raw, association_action_raw, raw_cu_loads = action
        x_i_raw = x_i_raw.flatten()
        association_action_raw = association_action_raw.flatten()

        x_i_raw = x_i_raw.astype(np.int32)
        association_action_raw = association_action_raw.astype(np.int32)

        if np.any(np.isnan(association_action_raw)) or np.any(np.isnan(x_i_raw)):
            association_action_raw = np.nan_to_num(
                association_action_raw, nan=0, 
                posinf=self.C - 1, neginf=0
            ).astype(np.int32)
            x_i_raw = np.nan_to_num(x_i_raw, nan=0).astype(np.int32)

        # SE-RL: Apply projections in sequence with penalty tracking
        projection_info = {
            'safety_deployments_added': 0,
            'validity_reassignments': 0,
            'nrf_violations_fixed': 0,
            'dus_moved_for_nrf': 0,
            'min_cus_required': self.min_cus_required,
            'actual_cus_deployed': 0,
            'deployment_changed': False,
            'association_changed': False,
            'final_cu_loads': np.zeros(self.C)
        }

        # 1. Safety projection (ensure minimum CUs deployed)
        x_i_projected = self.project_deployment_safety(x_i_raw)
        initial_deployed = np.sum(x_i_raw)
        final_deployed = np.sum(x_i_projected)
        projection_info['safety_deployments_added'] = final_deployed - initial_deployed
        projection_info['deployment_changed'] = (projection_info['safety_deployments_added'] > 0)
        projection_info['actual_cus_deployed'] = final_deployed

        # 2. Validity projection (ensure only deployed CUs are associated)
        association_action_valid = self.project_association_validity(association_action_raw, x_i_projected)
        validity_changes = np.sum(association_action_raw != association_action_valid)
        projection_info['validity_reassignments'] = validity_changes
        projection_info['association_changed'] = (validity_changes > 0)

        # 3. NRF projection (respect capacity constraints)
        association_action_projected, nrf_info = self.project_nrf_constraints(association_action_valid, x_i_projected)
        projection_info.update(nrf_info)
        
        # Final association changes
        total_association_changes = np.sum(association_action_raw != association_action_projected)
        projection_info['association_changed'] = (total_association_changes > 0)

        # Generate current traffic demands (already updated)
        current_demands = self.current_demands
        
        # Pure-Python port of the same MATLAB dispatcher, in legacy-physics mode.
        try:
            sinrs, rates = self.scn.dispatch(association_action_projected, self.PA)
            sinrs = np.asarray(sinrs).flatten()
            rates = np.asarray(rates).flatten()
            ncu = float(len(np.unique(association_action_projected)))
            debug_pt = 0
        except Exception as e:
            print(f"ERROR: Python physical dispatcher failed: {e}")
            sinrs = np.full(self.D, self.min_sinr)
            rates = np.zeros(self.D)  # Penalize with zero rates on error
            ncu = 0.0
            debug_pt = 1

        # Clean data
        sinrs = np.nan_to_num(sinrs, nan=self.min_sinr, posinf=self.max_sinr, neginf=self.min_sinr)
        rates = np.nan_to_num(rates, nan=0.0, posinf=10.0, neginf=0.0)

        self.rate_list = rates.copy()

        # LAGRANGIAN REWARD CALCULATION using projected actions WITH projection penalties
        reward, reward_components = self.calculate_lagrangian_reward(
            rates, current_demands, x_i_raw, association_action_raw, x_i_projected, association_action_projected, 
            projection_info['final_cu_loads'], projection_info
        )

        # Create next enhanced state
        next_obs = self.create_enhanced_state(rates, current_demands)
        
        # Update tracking
        self.prev_rates = rates.copy()
        self.prev_demands = current_demands.copy()
        self.timestep += 1

        # Done condition
        qos_satisfaction = self.calculate_qos_satisfaction(rates, current_demands)
        terminated = (qos_satisfaction > 0.999 and reward_components['rate_viol'] <= 0.0000001)

        # SE-RL: Enhanced info with projection statistics
        info = {
            'rates': rates,
            'traffic_demands': current_demands,
            'ncu': ncu,
            'rate_viol': reward_components['rate_viol'],
            'nrf_viol': reward_components['nrf_viol'],
            'qos_satisfaction': qos_satisfaction,
            'spectral_efficiency': reward_components['spectral_efficiency'],
            'reward_components': reward_components,
            # SE-RL specific: projection information
            'projection_info': projection_info,
            'raw_actions': {
                'x_i_raw': x_i_raw,
                'association_raw': association_action_raw,
                'raw_cu_loads': raw_cu_loads
            },
            'projected_actions': {
                'x_i_projected': x_i_projected,
                'association_projected': association_action_projected,
                'projected_cu_loads': projection_info['final_cu_loads']
            }
        }
        
        return next_obs, reward, terminated, info

    def calculate_lagrangian_reward(self, rates: np.ndarray, demands: np.ndarray, x_i_raw: np.ndarray, association_action_raw: np.ndarray, 
                                   x_i: np.ndarray, association_action: np.ndarray, cu_loads: np.ndarray,
                                   projection_info: dict) -> Tuple[float, dict]:
        """
        SE-RL: Calculate Lagrangian reward WITH projection penalties
        L = Σ rates - λ_s Σ x_i + λ_2 Σ g_2i + λ_3 Σ g_3j - PROJECTION_PENALTIES
        
        The policy must learn to avoid projections to maximize reward!
        """
        
        # Main objective: Σ rates - λ_s Σ deployments
        sum_rates = np.sum(rates)
        deployment_cost = self.lambda_sparsity * np.sum(x_i)
        main_objective = sum_rates - deployment_cost
        print(f"rates = {rates}")
        
        
        # Constraint functions (≤ 0 form) - computed on PROJECTED actions
        # g_2i = load_i - N_RF * x_i ≤ 0 (NRF constraint).
        # The legacy path evaluates this after NRF projection, making it
        # identically zero. T5 can evaluate the same formulated constraint on
        # the raw policy action, while the default remains provenance-faithful.
        if os.environ.get("CHARM_RAW_NRF_LAGRANGIAN", "0") == "1":
            raw_loads = np.bincount(
                association_action_raw.astype(np.int64), minlength=self.C
            )[:self.C]
            g2_temp = np.maximum(0.0, raw_loads - self.NRF * x_i_raw)
        else:
            g2_temp = np.maximum(0.0, cu_loads - self.NRF * x_i)
        g2_viol = -g2_temp  # Negative excess
        
        # g_3j = min(R_min, d_j) - rates_j ≤ 0 (Rate constraint)
        min_required_rates = np.minimum(self.curr_min_rate, demands)
        g3_temp = np.maximum(0.0, min_required_rates - rates)  # Shortfall
        g3_viol = -g3_temp  # Negative shortfall (proportional)
        
        # Lagrangian terms: +λ_k * g_k (positive multipliers, negative for violations)
        nrf_lagrangian = self.lambda_2 * np.sum(g2_viol)
        rate_lagrangian = self.lambda_3 * np.sum(g3_viol)
        
        # SE-RL: NETWORK-SCALED PROJECTION PENALTIES - Policy must learn to avoid these!
        projection_penalties = 0.0
        penalties = self.penalty_multipliers
        
        # Penalty for safety deployments (policy should deploy at least min_cus_required CUs)
        safety_penalty = -penalties['safety'] * projection_info['safety_deployments_added']
        projection_penalties += safety_penalty
        #print(f"safety_penalty = {safety_penalty}")
        
        
        # Penalty for association validity violations (policy should only associate to deployed CUs)
        validity_penalty = -penalties['validity'] * projection_info['validity_reassignments']  
        projection_penalties += validity_penalty
        #print(f"validity_penalty = {validity_penalty}")
        
        # Penalty for NRF violations (policy should respect NRF constraints)
        nrf_violation_penalty = -penalties['nrf_violation'] * projection_info['nrf_violations_fixed']
        dus_moved_penalty = -penalties['movement'] * projection_info['dus_moved_for_nrf']
        projection_penalties += (nrf_violation_penalty + dus_moved_penalty)    # dont forget minus sign. Rest all are plus but since we are multiplying we need to be careful here
        
        #print(f"nrfviolation_penaltyxdus_moved = {(nrf_violation_penalty * dus_moved_penalty)}")
        
        # Total Lagrangian reward WITH projection penalties
        total_lagrangian = main_objective + nrf_lagrangian + rate_lagrangian + projection_penalties
        
        #print(f"main_objective = {main_objective}")
        
        print(f"nrf_lagrangian = {nrf_lagrangian}")
        
        print(f"rate_lagrangian = {rate_lagrangian}")
        
        print(f"projection_penalties = {projection_penalties}")
        
        
        
        # Association constraint (penalty for invalid associations - should be zero after projection)
        invalid_associations = np.sum(x_i[association_action] == 0)
        if np.sum(x_i_raw)==0:
            invalid_associations_raw = np.sum(x_i_raw[association_action_raw] == 0)
            association_penalty_raw_leftover = -penalties['validity'] * (invalid_associations_raw - projection_info['validity_reassignments'])
            total_lagrangian += association_penalty_raw_leftover   # projection takes off some invalid associations especially when zero deployment
        
        # For backward compatibility - true violation metrics
        rate_viol = np.mean(g3_temp)   # Average shortfall per DU
        nrf_viol = np.sum(g2_temp) / (np.sum(x_i) + 1e-8)   # Average excess per deployed CU
        
        # Spectral efficiency (for compatibility)
        spectral_efficiency = sum_rates / (np.sum(x_i) + 1e-8)
        spectral_efficiency_norm = np.tanh(spectral_efficiency / 10.0)

        ## If no deployment make reward more negative
        if np.sum(x_i_raw)==0:
            no_deployment_penalty = -100
            total_lagrangian +=no_deployment_penalty
        
        # Clip to reasonable range
        total_lagrangian = np.clip(total_lagrangian, -1000.0, 70.0)

        reward_components = {
            'sum_rates': sum_rates,
            'deployment_cost': deployment_cost,
            'rate_viol': rate_viol,  # Average shortfall
            'nrf_viol': nrf_viol,    # Average excess
            'spectral_efficiency': spectral_efficiency_norm,
            'association_violations': invalid_associations / self.D,  # Violation fraction
            'main_objective': main_objective,
            'nrf_lagrangian': nrf_lagrangian,
            'rate_lagrangian': rate_lagrangian,           
            'g2_constraints': g2_viol,  # Raw constraint values
            'g3_constraints': g3_viol,  # Raw constraint values
            # SE-RL specific: network-scaled projection penalties
            'projection_penalties': projection_penalties,
            'safety_penalty': safety_penalty,
            'validity_penalty': validity_penalty, 
            'nrf_violation_penalty': nrf_violation_penalty,
            'dus_moved_penalty': dus_moved_penalty,
            # Network scaling metadata
            'penalty_scale_factor': penalties['scale_factor'],
            'network_scale': penalties['network_scale'],
            'expected_penalty_obj_ratio': penalties['expected_penalty_obj_ratio'],
            'actual_penalty_obj_ratio': abs(projection_penalties) / (abs(main_objective) + 1e-8)
        }
        
        return float(total_lagrangian), reward_components

    def calculate_qos_satisfaction(self, rates: np.ndarray, demands: np.ndarray) -> float:
        """QoS satisfaction based on constraint r_j ≥ min(R_min, d_j)"""
        min_required_rates = np.minimum(self.curr_min_rate, demands)
        satisfied = np.sum(rates >= min_required_rates * 0.9)
        return satisfied / self.D
    
    def get_num_cus_pertimestep(self, association_action: np.ndarray) -> int:
        """Get number of unique CUs used"""
        return len(np.unique(association_action))
    
    def close(self) -> None:
        """Clean up environment"""
        pass
