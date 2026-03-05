import pandas as pd 
import yfinance as yf  
import pandas_ta as ta
import pandas_datareader.data as web
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Beta
import matplotlib.pyplot as plt

# ============================================================================
# GPU SETUP
# ============================================================================

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"🚀 GPU ENABLED: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🚀 Apple Silicon GPU (MPS) ENABLED")
else:
    device = torch.device("cpu")
    print("⚠️  Running on CPU")

print(f"   Device: {device}\n")

# ============================================================================
# DATA ENGINE
# ============================================================================

class PortfolioDataEngine:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start = start_date
        self.end = end_date
    
    def fetch_portfolio_data(self):
        print(f"Fetching data for {len(self.tickers)} assets...")
        
        all_data = {}
        for ticker in self.tickers:
            print(f"  {ticker}...", end=" ")
            df = yf.download(ticker, start=self.start, end=self.end, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            df['Returns'] = df['Close'].pct_change()
            df['SMA_20'] = ta.sma(df['Close'], length=20)
            df['SMA_50'] = ta.sma(df['Close'], length=50)
            df['RSI'] = ta.rsi(df['Close'], length=14)
            
            all_data[ticker] = df
            print("✓")
        
        base = self.tickers[0]
        portfolio_df = all_data[base][['Close']].copy()
        portfolio_df.columns = [f'{base}_Close']
        
        for ticker in self.tickers:
            for col in ['Close', 'Returns', 'RSI']:
                if col in all_data[ticker].columns:
                    portfolio_df[f'{ticker}_{col}'] = all_data[ticker][col]
        
        portfolio_df = portfolio_df.ffill().bfill().dropna()
        print(f"Data shape: {portfolio_df.shape}\n")
        
        return portfolio_df, all_data


# ============================================================================
# HOLD-OR-REBALANCE ENVIRONMENT
# ============================================================================

class HoldRebalanceEnv:
    """
    Agent outputs:
    - action_type: 0=HOLD, 1=REBALANCE  
    - weights: [w1, w2, ..., wn] (only used if action_type=1)
    """
    
    def __init__(self, portfolio_data, tickers, initial_balance=100000):
        self.data = portfolio_data.reset_index(drop=True)
        self.tickers = tickers
        self.n_assets = len(tickers)
        self.initial_balance = initial_balance
        
        # CRITICAL: Reduce transaction costs
        self.commission = 0.0002  # 0.02% (was 0.1%)
        self.rebalance_cost = 0.0001  # 0.01% (was 0.05%)
        
        self.normalize_features()
        self.reset()
    
    def normalize_features(self):
        self.norm_features = []
        
        for ticker in self.tickers:
            for suffix in ['Returns', 'RSI']:
                col = f'{ticker}_{suffix}'
                if col in self.data.columns:
                    mean, std = self.data[col].mean(), self.data[col].std()
                    self.data[f'{col}_norm'] = (self.data[col] - mean) / (std + 1e-8)
                    self.norm_features.append(f'{col}_norm')
    
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        
        # Start with equal weights
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.shares = np.zeros(self.n_assets)
        
        # Initial purchase
        prices = self._get_prices()
        for i in range(self.n_assets):
            allocation = self.balance / self.n_assets
            shares = allocation // (prices[i] * (1 + self.commission))
            cost = shares * prices[i] * (1 + self.commission)
            self.shares[i] = shares
            self.balance -= cost
        
        self.portfolio_history = [self.initial_balance]
        self.rebalance_history = []
        self.rebalance_count = 0
        self.days_since_rebalance = 0
        
        return self._get_state()
    
    def _get_state(self):
        if self.current_step >= len(self.data):
            return np.zeros(2 * self.n_assets + len(self.norm_features) + 3, dtype=np.float32)
        
        row = self.data.iloc[self.current_step]
        
        # Current weights
        state = list(self.weights)
        
        # Returns for each asset (raw, not normalized - important signal!)
        for ticker in self.tickers:
            col = f'{ticker}_Returns'
            if col in row.index:
                state.append(row[col] if not np.isnan(row[col]) else 0.0)
        
        # Normalized features
        for col in self.norm_features:
            if col in row.index:
                state.append(row[col] if not np.isnan(row[col]) else 0.0)
        
        # Portfolio metrics
        state.append((self.portfolio_value / self.initial_balance) - 1)  # Total return
        state.append(self.days_since_rebalance / 100.0)  # Normalized days
        state.append(np.sum(self.weights ** 2))  # Concentration
        
        return np.array(state, dtype=np.float32)
    
    def step(self, action_type, weights):
        """
        action_type: 0=HOLD, 1=REBALANCE
        weights: target portfolio weights (only used if rebalancing)
        """
        if self.current_step >= len(self.data) - 1:
            return self._get_state(), 0, True, {}
        
        # FIXED: Calculate portfolio value at START of day (before any action)
        prices = self._get_prices()
        holdings_value = np.sum(self.shares * prices)
        old_value = self.balance + holdings_value
        
        did_rebalance = False
        
        if action_type == 1:  # REBALANCE
            # Normalize weights
            weights = np.abs(weights)
            weights = weights / (weights.sum() + 1e-8)
            
            # Sell everything (at current prices)
            self.balance = old_value
            self.shares = np.zeros(self.n_assets)
            
            # Rebalancing cost (flat fee)
            rebalance_cost = self.balance * self.rebalance_cost
            self.balance -= rebalance_cost
            
            # Buy new allocation (at current prices)
            for i, w in enumerate(weights):
                if w > 0.001:
                    allocation = self.balance * w
                    shares = allocation // (prices[i] * (1 + self.commission))
                    cost = shares * prices[i] * (1 + self.commission)
                    self.shares[i] = shares
                    self.balance -= cost
            
            self.weights = weights
            self.rebalance_count += 1
            self.rebalance_history.append(self.current_step)
            self.days_since_rebalance = 0
            did_rebalance = True
        
        # Move to next day
        self.current_step += 1
        self.days_since_rebalance += 1
        
        # FIXED: Calculate portfolio value at END of day (next day's prices)
        next_prices = self._get_prices()
        holdings_value = np.sum(self.shares * next_prices)
        new_value = self.balance + holdings_value
        self.portfolio_value = new_value
        
        # Update actual weights based on price movements
        if self.portfolio_value > 0:
            self.weights = (self.shares * next_prices) / self.portfolio_value
        
        # REWARD: Daily return from old_value to new_value
        daily_return = (new_value - old_value) / old_value
        reward = daily_return * 100  # Scale to percentage
        
        # Penalty for rebalancing
        if did_rebalance:
            reward -= 0.05  # Increased penalty to discourage overtrading
        
        self.portfolio_history.append(self.portfolio_value)
        done = self.current_step >= len(self.data) - 1
        
        info = {
            'portfolio_value': self.portfolio_value,
            'did_rebalance': did_rebalance,
            'rebalances': self.rebalance_count
        }
        
        return self._get_state(), reward, done, info
    
    def _get_prices(self):
        prices = []
        for ticker in self.tickers:
            prices.append(self.data.iloc[self.current_step][f'{ticker}_Close'])
        return np.array(prices)
    
    def get_state_dim(self):
        return len(self._get_state())


# ============================================================================
# ACTOR-CRITIC NETWORK
# ============================================================================

class HoldRebalanceActorCritic(nn.Module):
    def __init__(self, state_dim, n_assets):
        super().__init__()
        
        hidden = 256
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        
        # Action head: HOLD vs REBALANCE (biased toward HOLD)
        self.action_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Logits for [HOLD, REBALANCE]
        )
        
        # Initialize with bias toward HOLD
        self.action_head[-1].bias.data[0] = 2.0  # HOLD
        self.action_head[-1].bias.data[1] = -2.0  # REBALANCE
        
        # Weights head: portfolio allocation (only used if REBALANCE chosen)
        self.weights_alpha = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, n_assets),
            nn.Softplus()
        )
        
        self.weights_beta = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, n_assets),
            nn.Softplus()
        )
        
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.to(device)
    
    def forward(self, state):
        features = self.encoder(state)
        
        action_logits = self.action_head(features)
        alpha = self.weights_alpha(features) + 2.0  # Higher concentration
        beta = self.weights_beta(features) + 2.0
        value = self.critic(features)
        
        return action_logits, alpha, beta, value


# ============================================================================
# PPO AGENT
# ============================================================================

class HoldRebalancePPO:
    def __init__(self, state_dim, n_assets, lr=5e-5):
        self.policy = HoldRebalanceActorCritic(state_dim, n_assets)
        self.policy_old = HoldRebalanceActorCritic(state_dim, n_assets)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.K_epochs = 4
        
        self.memory = {
            'states': [],
            'action_types': [],
            'weights': [],
            'action_log_probs': [],
            'weight_log_probs': [],
            'rewards': [],
            'dones': []
        }
    
    def select_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action_logits, alpha, beta, _ = self.policy_old(state_t)
            
            # Sample action type
            action_dist = Categorical(logits=action_logits)
            action_type = action_dist.sample()
            action_log_prob = action_dist.log_prob(action_type)
            
            # Sample weights (always sample, but only use if REBALANCE)
            weight_dist = Beta(alpha, beta)
            raw_weights = weight_dist.sample()
            weights = raw_weights / raw_weights.sum(dim=-1, keepdim=True)
            weight_log_prob = weight_dist.log_prob(raw_weights).sum(dim=-1)
        
        self.memory['states'].append(state_t.cpu())
        self.memory['action_types'].append(action_type.cpu())
        self.memory['weights'].append(raw_weights.cpu())
        self.memory['action_log_probs'].append(action_log_prob.cpu())
        self.memory['weight_log_probs'].append(weight_log_prob.cpu())
        
        return action_type.item(), weights.cpu().numpy()[0]
    
    def store_reward_done(self, reward, done):
        self.memory['rewards'].append(reward)
        self.memory['dones'].append(done)
    
    def update(self):
        if len(self.memory['rewards']) == 0:
            return
        
        # Compute returns
        returns = []
        R = 0
        for r, done in zip(reversed(self.memory['rewards']), reversed(self.memory['dones'])):
            if done:
                R = 0
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)
        
        old_states = torch.cat(self.memory['states']).to(device)
        old_action_types = torch.cat(self.memory['action_types']).to(device)
        old_weights = torch.cat(self.memory['weights']).to(device)
        old_action_lps = torch.cat(self.memory['action_log_probs']).to(device)
        old_weight_lps = torch.cat(self.memory['weight_log_probs']).to(device)
        
        for _ in range(self.K_epochs):
            action_logits, alpha, beta, values = self.policy(old_states)
            values = values.squeeze()
            
            # Action type loss
            action_dist = Categorical(logits=action_logits)
            action_lps = action_dist.log_prob(old_action_types)
            action_entropy = action_dist.entropy()
            
            # Weight loss
            weight_dist = Beta(alpha, beta)
            weight_lps = weight_dist.log_prob(old_weights).sum(dim=-1)
            weight_entropy = weight_dist.entropy().sum(dim=-1)
            
            # Combined log prob
            total_lps = action_lps + weight_lps
            old_total_lps = old_action_lps + old_weight_lps
            
            # PPO loss
            ratios = torch.exp(total_lps - old_total_lps.detach())
            advantages = returns - values.detach()
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * ((values - returns) ** 2).mean()
            entropy_bonus = -0.001 * (action_entropy.mean() + weight_entropy.mean())
            
            loss = actor_loss + critic_loss + entropy_bonus
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        for key in self.memory:
            self.memory[key] = []
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()


# ============================================================================
# TRAINING
# ============================================================================

def train_hold_rebalance(tickers, start="2018-01-01", end="2024-01-01", episodes=100):
    print("="*70)
    print("HOLD vs REBALANCE RL TRAINING")
    print("="*70 + "\n")
    
    data_engine = PortfolioDataEngine(tickers, start, end)
    portfolio_data, asset_data = data_engine.fetch_portfolio_data()
    
    env = HoldRebalanceEnv(portfolio_data, tickers)
    agent = HoldRebalancePPO(env.get_state_dim(), len(tickers))
    
    episode_returns = []
    
    print(f"State dim: {env.get_state_dim()}")
    print(f"Assets: {len(tickers)}")
    print(f"Data points: {len(env.data)}\n")
    
    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0
        
        while True:
            action_type, weights = agent.select_action(state)
            state, reward, done, info = env.step(action_type, weights)
            agent.store_reward_done(reward, done)
            ep_reward += reward
            
            if done:
                break
        
        agent.update()
        
        final_value = env.portfolio_value
        total_return = ((final_value - env.initial_balance) / env.initial_balance) * 100
        episode_returns.append(total_return)
        
        if (ep + 1) % 10 == 0:
            avg10 = np.mean(episode_returns[-10:])
            returns = np.diff(env.portfolio_history) / env.portfolio_history[:-1]
            sharpe = (np.mean(returns) / (np.std(returns) + 1e-8)) * np.sqrt(252)
            
            # Calculate hold vs rebalance ratio
            total_steps = len(env.data) - 1
            rebalance_pct = (env.rebalance_count / total_steps) * 100
            
            print(f"Ep {ep+1:3d} | Return: {total_return:6.2f}% | Sharpe: {sharpe:5.2f} | "
                  f"Avg10: {avg10:6.2f}% | Rebal: {env.rebalance_count:4d} ({rebalance_pct:.1f}%) | "
                  f"Value: ${final_value:,.0f}")
    
    # Calculate buy & hold for comparison
    print(f"\n{'='*70}")
    print("BUY & HOLD COMPARISON")
    print(f"{'='*70}")
    
    equal_returns = []
    for ticker in tickers:
        data = asset_data[ticker]
        start_price = data['Close'].iloc[0]
        end_price = data['Close'].iloc[-1]
        ret = ((end_price - start_price) / start_price) * 100
        equal_returns.append(ret)
        print(f"{ticker:6s}: {ret:6.2f}%")
    
    equal_weight = np.mean(equal_returns)
    print(f"\nEqual Weight Portfolio: {equal_weight:.2f}%")
    print(f"RL Agent Final Return: {episode_returns[-1]:.2f}%")
    print(f"Outperformance: {episode_returns[-1] - equal_weight:+.2f}%")
    
    return agent, env, episode_returns


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    agent, env, returns = train_hold_rebalance(
        tickers=TICKERS,
        start="2018-01-01",
        end="2024-01-01",
        episodes=100
    )
    
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"Final Episode Return: {returns[-1]:.2f}%")
    print(f"Best Episode Return: {max(returns):.2f}%")
    print(f"Average Return (last 20): {np.mean(returns[-20:]):.2f}%")
    print(f"{'='*70}")