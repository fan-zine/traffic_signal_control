import torch
import torch.optim as optim
from torch.distributions.categorical import Categorical
from ..models.util import process_observation_buffer_with_graph

class PGSingleAgent:
    def __init__(self, actor, k, ts_idx, device, num_nodes, max_lanes, lr=1e-3):
        self.hid_dim = actor.encoder._hid_dim
        self.actor = actor  # encoder + head
        self.ts_idx = ts_idx
        self.device = device
        self.num_nodes = num_nodes
        self.max_lanes = max_lanes
        self.optimizer = optim.Adam(list(actor.parameters()), lr=lr)
        self.log_probs = []
        self.rewards = []
        self.last_k_observations = []
        self.k = k

    def process_observations(self):
        processed_obs = process_observation_buffer_with_graph(
            self.last_k_observations, self.ts_idx, self.max_lanes, self.num_nodes,
        )
        return torch.tensor(processed_obs, dtype=torch.float32).to(self.device) # shape of (seq_len, num_nodes, input_dim)

    def select_actions(self, logits):
        """
        Select actions by sampling from the policy distribution.

        Args:
            logits (torch.Tensor): Action logits of shape (num_ts, max_green_phases).

        Returns:
            actions (dict): {ts_id: greenPhase}.
        """
        action_distributions = torch.softmax(logits, dim=-1)

        actions = {}
        log_prob_sum = 0
        for ts_id, node_idx in self.ts_idx.items():
            dist = Categorical(action_distributions[node_idx])
            action = dist.sample()
            actions[ts_id] = action.item()
            log_prob_sum += dist.log_prob(action)  # Sum log probabilities across all traffic signals

        self.log_probs.append(log_prob_sum)

        return actions

    def compute_global_reward(self, rewards):
        """
        Compute global reward as the sum of rewards from all traffic lights.
        """
        return sum(rewards.values())

    def train(self, env, num_episodes):
        for episode in range(num_episodes):
            # Reset environment and observation buffer
            obs = env.reset()
            env.fixed_ts = True
            self.log_probs = []
            self.rewards = []
            self.last_k_observations = [obs]
            step_count = 0

            while True:
                # Warmup phase with default traffic light control
                if step_count < self.k:
                    obs, _, _, _ = env.step(action=None)  # Run default traffic light control
                    self.last_k_observations.append(obs)
                    step_count += 1
                    continue

                # RL control starts after warmup
                if step_count == self.k:
                    env.fixed_ts = False  # Switch to RL control
                    for _, ts in env.traffic_signals.items():
                        ts.run_rl_agents()  # Activate RL agents

                #output_features = self.encoder(obs, initial_hidden_state)  # (1, num_nodes, hid_dim)
                #input_features = obs[-1]  # (1, num_nodes, input_dim)
                #logits = self.predictor(output_features, input_features).squeeze(0)  # (num_ts, max_green_phases)
                #print(f"logits: {logits}")

                obs = self.process_observations()
                obs = obs.unsqueeze(1)  # Shape: (seq_len, 1, num_nodes, input_dim)
                initial_hidden_state = torch.zeros((1, self.num_nodes * self.hid_dim), device=self.device)
                logits = self.actor(obs, initial_hidden_state).squeeze(0)

                actions = self.select_actions(logits)
                obs, rewards, dones, infos = env.step(actions)
                self.last_k_observations.append(obs)
                if len(self.last_k_observations) > self.k:
                    self.last_k_observations.pop(0)

                global_reward = self.compute_global_reward(rewards)
                self.rewards.append(global_reward)

                if dones["__all__"]:
                    break

                step_count += 1

            # Compute policy loss and update
            discounted_rewards = self._compute_discounted_rewards()
            policy_loss = self._compute_policy_loss(discounted_rewards)

            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

            print(f"Episode {episode}, Total Reward: {sum(self.rewards)}, Loss: {policy_loss.item()}")

        env.save_csv(env.out_csv_name, num_episodes)
        env.close()

    def _compute_discounted_rewards(self, gamma=0.99):
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(self.rewards):
            cumulative_reward = reward + gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(self.device)
        return (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)

    def _compute_policy_loss(self, discounted_rewards):
        policy_loss = 0
        for log_prob, reward in zip(self.log_probs, discounted_rewards):
            policy_loss -= log_prob * reward
        return policy_loss

    def save_models(self, model_path):
        """
        Save trained model weights.
        """
        torch.save(self.actor.state_dict(), model_path)

    def test(self, env):
        """
        Test the RL agent.

        Args:
            env: SumoEnvironment.
            num_episodes (int): Number of episodes to run.
        """

        obs = env.reset()
        env.fixed_ts = True
        self.last_k_observations = [obs]
        step_count = 0

        while True:
            # Warmup phase with default traffic light control
            if step_count < self.k:
                obs, _, _, _ = env.step(action=None)  # Run default traffic light control
                self.last_k_observations.append(obs)
                step_count += 1
                continue

            # RL control starts after warmup
            if step_count == self.k:
                env.fixed_ts = False  # Switch to RL control
                for _, ts in env.traffic_signals.items():
                    ts.run_rl_agents()  # Activate RL agents

            with torch.no_grad:
                obs = self.process_observations()
                obs = obs.unsqueeze(1)  # Shape: (seq_len, 1, num_nodes, input_dim)
                initial_hidden_state = torch.zeros((1, self.num_nodes * self.hid_dim), device=self.device)
                logits = self.actor(obs, initial_hidden_state).squeeze(0)
                actions = self.select_actions(logits)

            obs, rewards, dones, infos = env.step(actions)
            self.last_k_observations.append(obs)
            if len(self.last_k_observations) > self.k:
                self.last_k_observations.pop(0)

            if dones["__all__"]:
                break

            step_count += 1


