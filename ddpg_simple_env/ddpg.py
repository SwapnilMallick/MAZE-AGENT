"""
DDPG Implementation — 4-Room Maze Navigation
=============================================
Self-contained module following the Spinning Up DDPG specification.
Imports FourRoomMazeEnv from maze_env.py.

Networks
--------
  Actor  μ_θ  :  obs(4) → FC(256) → ReLU → FC(256) → ReLU → FC(2) → Tanh
  Critic Q_φ  :  (obs+act)(6) → FC(256) → ReLU → FC(256) → ReLU → FC(1)
  Targets: deep copies of main nets, updated by Polyak averaging (ρ=0.995)

Update order per gradient step (Spinning Up pseudocode)
--------------------------------------------------------
  1. Compute Bellman targets y using target networks   [torch.no_grad()]
  2. Critic update  : minimise MSE( Q_φ(s,a), y )     [∇_φ only]
  3. Actor  update  : maximise Q_φ(s, μ_θ(s))         [∇_θ only, critic frozen]
  4. Polyak update  : nudge targets toward mains       [no gradients]

Replay buffer
-------------
  Circular array of capacity 100 000.
  Stores terminated (not done=terminated|truncated) so truncated episodes
  are correctly bootstrapped in the Bellman target.
"""

import copy
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# Allow running from any working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maze_env import FourRoomMazeEnv, EmptyGridEnv


# ─────────────────────────────────────────────────────────────────────────────
#  Networks
# ─────────────────────────────────────────────────────────────────────────────

class Actor(nn.Module):
    """
    Deterministic policy  μ_θ(s) : obs → action ∈ (−1, +1)^act_dim

    Tanh at the output layer ensures actions are bounded without any
    explicit clipping inside the network.  Exploration noise is added
    externally in select_action(), after which we clip once more.
    """

    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256)):
        super().__init__()
        layers = []
        in_sz  = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_sz, h), nn.ReLU()]
            in_sz = h
        layers += [nn.Linear(in_sz, act_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        # obs : (batch, obs_dim)  →  (batch, act_dim)
        return self.net(obs)


class Critic(nn.Module):
    """
    Action-value function  Q_φ(s, a) : (obs, act) → scalar

    State and action are concatenated at the input layer (6-D for our
    problem).  No output activation — Q-values are unbounded scalars.
    """

    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256)):
        super().__init__()
        layers = []
        in_sz  = obs_dim + act_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_sz, h), nn.ReLU()]
            in_sz = h
        layers += [nn.Linear(in_sz, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, obs, act):
        # obs : (batch, obs_dim),  act : (batch, act_dim)
        x = torch.cat([obs, act], dim=-1)
        return self.net(x).squeeze(-1)   # → (batch,)


# ─────────────────────────────────────────────────────────────────────────────
#  Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Fixed-capacity circular buffer.
    Oldest entries are overwritten once capacity is reached.

    Stores `terminated` (not `terminated | truncated`) as the done flag.
    Rationale: a truncated episode (max-steps exceeded) is NOT a true
    terminal state — the value at s′ should still be bootstrapped.
    Using truncated=True as done would incorrectly zero out the bootstrap
    term for those transitions.
    """

    def __init__(self, obs_dim, act_dim, capacity):
        self.capacity = capacity
        self.ptr      = 0       # write pointer
        self.size     = 0       # current number of stored transitions

        self.obs  = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act  = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew  = np.zeros( capacity,            dtype=np.float32)
        self.obs_ = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros( capacity,            dtype=np.float32)  # 0/1

    def store(self, obs, act, rew, obs_, terminated):
        """Write one transition; advance and wrap the write pointer."""
        self.obs [self.ptr] = obs
        self.act [self.ptr] = act
        self.rew [self.ptr] = rew
        self.obs_[self.ptr] = obs_
        self.done[self.ptr] = float(terminated)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        """Draw a uniform random mini-batch; return tensors on `device`."""
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            'obs':  torch.as_tensor(self.obs [idx], device=device),
            'act':  torch.as_tensor(self.act [idx], device=device),
            'rew':  torch.as_tensor(self.rew [idx], device=device),
            'obs_': torch.as_tensor(self.obs_[idx], device=device),
            'done': torch.as_tensor(self.done[idx], device=device),
        }

    def __len__(self):
        return self.size


# ─────────────────────────────────────────────────────────────────────────────
#  DDPG Agent
# ─────────────────────────────────────────────────────────────────────────────

class DDPGAgent:
    """
    DDPG agent: owns the four networks, optimisers, and the update logic.

    All tensors live on `device`.  The agent is framework-agnostic with
    respect to the environment — it only consumes numpy arrays and emits
    numpy arrays.
    """

    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes = (256, 256),
        actor_lr     = 1e-3,
        critic_lr    = 1e-3,
        gamma        = 0.98,
        polyak       = 0.995,
        act_noise    = 0.1,
        device       = 'cpu',
    ):
        self.gamma     = gamma
        self.polyak    = polyak
        self.act_noise = act_noise
        self.device    = torch.device(device)

        # ── Main networks ─────────────────────────────────────────────
        self.actor  = Actor (obs_dim, act_dim, hidden_sizes).to(self.device)
        self.critic = Critic(obs_dim, act_dim, hidden_sizes).to(self.device)

        # ── Target networks ───────────────────────────────────────────
        # Initialised as exact copies of main networks.
        # requires_grad=False: never receive gradient updates directly;
        # only moved via Polyak averaging in sub-step 4.
        self.actor_targ  = copy.deepcopy(self.actor ).to(self.device)
        self.critic_targ = copy.deepcopy(self.critic).to(self.device)

        for p in self.actor_targ.parameters():
            p.requires_grad = False
        for p in self.critic_targ.parameters():
            p.requires_grad = False

        # ── Optimisers ────────────────────────────────────────────────
        self.actor_opt  = Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_opt = Adam(self.critic.parameters(), lr=critic_lr)

    # ── Action selection ─────────────────────────────────────────────────────

    def select_action(self, obs, add_noise=True):
        """
        Compute μ_θ(obs), optionally adding Gaussian exploration noise.

        Parameters
        ----------
        obs       : np.ndarray, shape (obs_dim,)
        add_noise : bool  — True during training, False at test time

        Returns
        -------
        action : np.ndarray, shape (act_dim,), clipped to [−1, +1]
        """
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                    device=self.device).unsqueeze(0)
            act   = self.actor(obs_t).squeeze(0).cpu().numpy()

        if add_noise:
            act = act + self.act_noise * np.random.randn(*act.shape)

        return np.clip(act, -1.0, 1.0).astype(np.float32)

    # ── One gradient update ──────────────────────────────────────────────────

    def update(self, batch):
        """
        Execute one full DDPG gradient update over a mini-batch.

        Sub-step 1 — Compute Bellman targets y  [no gradient]
        Sub-step 2 — Critic update              [∇_φ]
        Sub-step 3 — Actor  update              [∇_θ, critic frozen]
        Sub-step 4 — Polyak update targets      [parameter interpolation]

        Returns
        -------
        (critic_loss, actor_loss) : (float, float)
        """
        obs  = batch['obs']
        act  = batch['act']
        rew  = batch['rew']
        obs_ = batch['obs_']
        done = batch['done']

        # ── Sub-step 1: Bellman targets ───────────────────────────────
        # target actor   → next action a′ = μ_θ_targ(s′)
        # target critic  → next value  Q_φ_targ(s′, a′)
        # y is detached from the computation graph — a fixed scalar per
        # transition that the main critic is trained to match.
        with torch.no_grad():
            next_act = self.actor_targ(obs_)
            q_targ   = self.critic_targ(obs_, next_act)
            y        = rew + self.gamma * (1.0 - done) * q_targ

        # ── Sub-step 2: Critic update ─────────────────────────────────
        # Loss: MSE between main critic Q_φ(s,a) and fixed target y.
        # Only φ (main critic) receives gradient updates here.
        self.critic_opt.zero_grad()
        q_pred      = self.critic(obs, act)
        critic_loss = F.mse_loss(q_pred, y)
        critic_loss.backward()
        self.critic_opt.step()

        # ── Sub-step 3: Actor update ──────────────────────────────────
        # Gradient flows:  μ_θ(s) → Q_φ(s, μ_θ(s)) → −mean → ∇_θ
        # Chain rule: ∂L/∂θ = (∂Q/∂a) · (∂μ/∂θ)
        #
        # The critic is frozen (requires_grad=False) so that PyTorch
        # does not compute or accumulate gradients w.r.t. φ during
        # this backward pass — they would be wasted and discarded.
        for p in self.critic.parameters():
            p.requires_grad = False

        self.actor_opt.zero_grad()
        actor_loss = -self.critic(obs, self.actor(obs)).mean()
        actor_loss.backward()
        self.actor_opt.step()

        # Restore critic gradients for the next critic update
        for p in self.critic.parameters():
            p.requires_grad = True

        # ── Sub-step 4: Polyak update ─────────────────────────────────
        # φ_targ ← ρ·φ_targ + (1−ρ)·φ
        # θ_targ ← ρ·θ_targ + (1−ρ)·θ
        # Runs after BOTH main network updates so targets absorb the
        # freshly updated φ and θ simultaneously.
        with torch.no_grad():
            for p, p_t in zip(self.actor.parameters(),
                               self.actor_targ.parameters()):
                p_t.data.mul_(self.polyak)
                p_t.data.add_((1.0 - self.polyak) * p.data)

            for p, p_t in zip(self.critic.parameters(),
                               self.critic_targ.parameters()):
                p_t.data.mul_(self.polyak)
                p_t.data.add_((1.0 - self.polyak) * p.data)

        return critic_loss.item(), actor_loss.item()

    # ── Checkpointing ────────────────────────────────────────────────────────

    def save(self, path):
        """Save all four network state dicts to a single file."""
        torch.save({
            'actor':       self.actor.state_dict(),
            'critic':      self.critic.state_dict(),
            'actor_targ':  self.actor_targ.state_dict(),
            'critic_targ': self.critic_targ.state_dict(),
        }, path)
        print(f"  [ckpt] saved → {path}")

    def load(self, path):
        """Restore all four network state dicts."""
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.actor_targ.load_state_dict(ckpt['actor_targ'])
        self.critic_targ.load_state_dict(ckpt['critic_targ'])


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(agent, env, n_episodes=10):
    """
    Run n_episodes deterministic episodes (no exploration noise).

    Returns
    -------
    dict with keys:
      mean_return   : float
      mean_length   : float
      success_rate  : float  (fraction of episodes where goal was reached)
    """
    returns, lengths, successes = [], [], []

    for _ in range(n_episodes):
        obs, _    = env.reset()
        ep_return = 0.0
        ep_len    = 0

        while True:
            act = agent.select_action(obs, add_noise=False)
            obs, rew, terminated, truncated, info = env.step(act)
            ep_return += rew
            ep_len    += 1

            if terminated or truncated:
                successes.append(float(info['goal_reached']))
                break

        returns.append(ep_return)
        lengths.append(ep_len)

    return {
        'mean_return':  float(np.mean(returns)),
        'mean_length':  float(np.mean(lengths)),
        'success_rate': float(np.mean(successes)),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Recording
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_and_record(agent, env_class, n_episodes, max_ep_len, results_dir, epoch=None):
    """
    Run n_episodes deterministic episodes, capture frames, and save one GIF
    per episode.  Returns the same stats dict as evaluate().

    A dedicated env instance with render_mode='rgb_array' is created and closed
    internally so the training env is never touched.

    Directory layout
    ----------------
    results_dir/
      epoch_0010/
        ep01_success_S(x,y)_G(x,y).gif
        ep02_failure_S(x,y)_G(x,y).gif
        ...

    Requires Pillow:  pip install Pillow
    """
    try:
        from PIL import Image as PILImage
    except ImportError:
        raise ImportError(
            "Pillow is required for GIF recording.  Install it with:\n"
            "    pip install Pillow"
        )

    epoch_label = f"epoch_{epoch:04d}" if epoch is not None else "final"
    ep_dir = os.path.join(results_dir, epoch_label)
    os.makedirs(ep_dir, exist_ok=True)

    env = env_class(max_steps=max_ep_len, render_mode='rgb_array')
    returns, lengths, successes = [], [], []

    for ep_idx in range(n_episodes):
        obs, _    = env.reset()
        start_pos = env.pos.copy()
        goal_pos  = env.goal.copy()

        ep_return = 0.0
        ep_len    = 0
        frames    = [env.render()]   # frame at t=0 (after reset)

        while True:
            act = agent.select_action(obs, add_noise=False)
            obs, rew, terminated, truncated, info = env.step(act)
            ep_return += rew
            ep_len    += 1
            frames.append(env.render())

            if terminated or truncated:
                successes.append(float(info['goal_reached']))
                break

        returns.append(ep_return)
        lengths.append(ep_len)

        outcome  = "success" if info['goal_reached'] else "failure"
        gif_name = (
            f"ep{ep_idx + 1:02d}_{outcome}"
            f"_S({start_pos[0]:.1f},{start_pos[1]:.1f})"
            f"_G({goal_pos[0]:.1f},{goal_pos[1]:.1f}).gif"
        )
        gif_path = os.path.join(ep_dir, gif_name)

        pil_frames = [PILImage.fromarray(f) for f in frames]
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=80,   # ms per frame  ≈ 12.5 FPS
            loop=0,
        )

    env.close()

    return {
        'mean_return':  float(np.mean(returns)),
        'mean_length':  float(np.mean(lengths)),
        'success_rate': float(np.mean(successes)),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    # ── Environment ──────────────────────────────────────
    env_class        = FourRoomMazeEnv,  # swap to EmptyGridEnv to validate algorithm
    max_ep_len       = 500,
    # ── Training scale ───────────────────────────────────
    epochs           = 100,
    steps_per_epoch  = 4_000,
    # ── Replay buffer ─────────────────────────────────────
    replay_size      = 100_000,
    batch_size       = 256,
    # ── Exploration schedule ──────────────────────────────
    start_steps      = 5_000,   # pure random actions before using policy
    update_after     = 1_000,   # gradient steps start after this many env steps
    update_every     = 1,       # gradient steps per env step (1:1 ratio)
    # ── Agent hyperparameters ─────────────────────────────
    hidden_sizes     = (256, 256),
    actor_lr         = 3e-4,
    critic_lr        = 3e-4,
    gamma            = 0.98,
    polyak           = 0.995,
    act_noise        = 0.1,
    # ── Reward shaping ────────────────────────────────────
    alpha            = 0.05,   # weight for distance-based shaping term r2
    # ── Evaluation & logging ──────────────────────────────
    n_eval_episodes  = 10,
    save_freq        = 10,      # checkpoint every N epochs
    checkpoint_dir   = 'checkpoints',
    results_dir      = 'results',
    log_file         = None,    # path for training log; None → checkpoint_dir/training_log.txt
    # ── Reproducibility ───────────────────────────────────
    seed             = 0,
    device           = 'cpu',
):
    """
    Full DDPG training loop on FourRoomMazeEnv.

    Structure (per environment step t)
    ------------------------------------
    1. If t ≤ start_steps  → uniform random action   (fills buffer)
       Else                → μ_θ(s) + Gaussian noise  (policy exploration)
    2. Execute action, observe (s′, r, terminated, truncated)
    3. Store (s, a, r, s′, terminated) in replay buffer
    4. If t > update_after, every update_every steps:
         sample batch → agent.update(batch) → log losses
    5. At epoch boundary:
         run evaluate() with no noise → log metrics → checkpoint if improved

    Returns
    -------
    agent : DDPGAgent  (trained)
    """

    # ── Reproducibility ───────────────────────────────────────────────────────
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Logging ───────────────────────────────────────────────────────────────
    _log_path = log_file or os.path.join(checkpoint_dir, 'training_log.txt')
    _log_f    = open(_log_path, 'w', buffering=1)   # line-buffered

    def log(msg=''):
        print(msg)
        _log_f.write(msg + '\n')

    # ── Environments ──────────────────────────────────────────────────────────
    train_env = env_class(max_steps=max_ep_len)
    eval_env  = env_class(max_steps=max_ep_len)

    obs_dim = train_env.observation_space.shape[0]   # 125 (4 scalars + 11×11 patch)
    act_dim = train_env.action_space.shape[0]        # 2

    # ── Agent & buffer ────────────────────────────────────────────────────────
    agent = DDPGAgent(
        obs_dim      = obs_dim,
        act_dim      = act_dim,
        hidden_sizes = hidden_sizes,
        actor_lr     = actor_lr,
        critic_lr    = critic_lr,
        gamma        = gamma,
        polyak       = polyak,
        act_noise    = act_noise,
        device       = device,
    )
    buffer = ReplayBuffer(obs_dim, act_dim, capacity=replay_size)
    device_t = torch.device(device)

    total_steps  = epochs * steps_per_epoch
    best_success = -1.0

    # ── Header ────────────────────────────────────────────────────────────────
    log("=" * 75)
    log(f"  DDPG — {env_class.__name__}")
    log(f"  obs_dim={obs_dim}  act_dim={act_dim}  device={device}")
    log(f"  epochs={epochs}  steps/epoch={steps_per_epoch}  "
        f"total={total_steps:,}")
    log(f"  start_steps={start_steps}  update_after={update_after}  "
        f"batch={batch_size}")
    log(f"  γ={gamma}  ρ={polyak}  σ_noise={act_noise}")
    log("=" * 75)
    log(f"  {'Epoch':>5} | {'Steps':>7} | {'Time':>6} | "
        f"{'TrRet':>7} | {'TrLen':>6} | {'TrSucc':>6} | "
        f"{'EvSucc':>6} | {'CrLoss':>7} | {'AcLoss':>7}")
    log("  " + "-" * 73)

    # ── Episode-level state ───────────────────────────────────────────────────
    obs, _    = train_env.reset(seed=seed)
    ep_return = 0.0
    ep_len    = 0

    # ── Per-epoch accumulators (cleared at each epoch boundary) ───────────────
    ep_returns:    list = []
    ep_lengths:    list = []
    ep_successes:  list = []
    critic_losses: list = []
    actor_losses:  list = []

    epoch   = 0
    t_start = time.time()

    # ── Main loop ─────────────────────────────────────────────────────────────
    for t in range(1, total_steps + 1):

        # ── Action selection ─────────────────────────────────────────────────
        if t <= start_steps:
            # Uniform random actions — ensures early buffer diversity before
            # any gradient step is taken.  Avoids training on nearly-identical
            # transitions from a randomly initialised policy.
            act = train_env.action_space.sample()
        else:
            act = agent.select_action(obs, add_noise=True)

        # ── Environment interaction ───────────────────────────────────────────
        old_pos = train_env.pos.copy()
        obs_, rew, terminated, truncated, info = train_env.step(act)

        old_dist = np.linalg.norm(old_pos - train_env.goal)
        new_dist = np.linalg.norm(train_env.pos - train_env.goal)
        r2 = alpha * (old_dist - new_dist)
        rew = rew + r2

        ep_return += rew
        ep_len    += 1

        # Store `terminated` (not terminated|truncated) as the done flag.
        # See ReplayBuffer docstring for rationale.
        buffer.store(obs, act, rew, obs_, terminated)
        obs = obs_

        # ── Episode reset ─────────────────────────────────────────────────────
        if terminated or truncated:
            ep_returns.append(ep_return)
            ep_lengths.append(ep_len)
            ep_successes.append(float(info['goal_reached']))
            ep_return = 0.0
            ep_len    = 0
            obs, _    = train_env.reset()

        # ── Gradient updates ──────────────────────────────────────────────────
        # Conditions: buffer must be warm (t > update_after) and enough
        # env steps have elapsed (t % update_every == 0).
        # Number of gradient steps per trigger = update_every (1:1 ratio).
        if t > update_after and t % update_every == 0:
            for _ in range(update_every):
                batch          = buffer.sample(batch_size, device_t)
                c_loss, a_loss = agent.update(batch)
                critic_losses.append(c_loss)
                actor_losses.append(a_loss)

        # ── Epoch boundary ────────────────────────────────────────────────────
        if t % steps_per_epoch == 0:
            epoch += 1
            elapsed = time.time() - t_start

            # Deterministic evaluation — record GIFs for the last 2 epochs only
            if epoch >= epochs - 1:
                eval_stats = evaluate_and_record(
                    agent, env_class, n_eval_episodes, max_ep_len, results_dir, epoch=epoch
                )
                log(f"  [rec] GIFs → {results_dir}/epoch_{epoch:04d}/")
            else:
                eval_stats = evaluate(agent, eval_env, n_episodes=n_eval_episodes)

            # Compute epoch stats (guard against empty accumulators)
            tr_ret   = np.mean(ep_returns)   if ep_returns   else float('nan')
            tr_len   = np.mean(ep_lengths)   if ep_lengths   else float('nan')
            tr_succ  = np.mean(ep_successes) if ep_successes else float('nan')
            c_loss_m = np.mean(critic_losses) if critic_losses else float('nan')
            a_loss_m = np.mean(actor_losses)  if actor_losses  else float('nan')

            log(
                f"  {epoch:5d} | {t:7,} | {elapsed:5.1f}s | "
                f"{tr_ret:+7.3f} | {tr_len:6.1f} | {tr_succ:6.3f} | "
                f"{eval_stats['success_rate']:6.3f} | "
                f"{c_loss_m:7.4f} | {a_loss_m:+7.4f}"
            )

            # Checkpoint best model (by eval success rate)
            if eval_stats['success_rate'] > best_success:
                best_success = eval_stats['success_rate']
                agent.save(os.path.join(checkpoint_dir, 'best.pt'))

            # Periodic checkpoint
            if epoch % save_freq == 0:
                agent.save(
                    os.path.join(checkpoint_dir, f'epoch_{epoch:04d}.pt')
                )

            # Clear per-epoch accumulators
            ep_returns.clear()
            ep_lengths.clear()
            ep_successes.clear()
            critic_losses.clear()
            actor_losses.clear()

    # ── Final save ────────────────────────────────────────────────────────────
    agent.save(os.path.join(checkpoint_dir, 'final.pt'))
    log(f"\n  Training complete.  Best eval success rate: {best_success:.3f}")
    _log_f.close()
    return agent


# ─────────────────────────────────────────────────────────────────────────────
#  Smoke test  (runs without full training)
# ─────────────────────────────────────────────────────────────────────────────

def run_smoke_test():
    """
    Verify all components are correctly wired without running full training.

    Checks:
      1. Network output shapes
      2. Critic and actor gradient isolation
      3. Polyak update correctness
      4. Replay buffer store / sample
      5. One full agent.update() call
      6. Select_action with and without noise
      7. evaluate() runs without error
    """
    print("=" * 60)
    print("  DDPG Smoke Test")
    print("=" * 60)

    # Read dimensions from a live environment instance so the smoke test
    # automatically adapts when the observation space changes (e.g. patch size).
    _env_tmp = FourRoomMazeEnv()
    OBS_DIM  = _env_tmp.observation_space.shape[0]   # 125 (4 scalars + 11×11 patch)
    ACT_DIM  = _env_tmp.action_space.shape[0]        # 2
    _env_tmp.close()

    BATCH    = 16
    CAPACITY = 500

    agent  = DDPGAgent(OBS_DIM, ACT_DIM)
    buffer = ReplayBuffer(OBS_DIM, ACT_DIM, capacity=CAPACITY)
    env    = FourRoomMazeEnv()
    device = torch.device('cpu')

    # ── [1] Network output shapes ─────────────────────────────────────
    print("\n[1] Network output shapes")
    dummy_obs = torch.zeros(BATCH, OBS_DIM)
    dummy_act = torch.zeros(BATCH, ACT_DIM)
    actor_out  = agent.actor(dummy_obs)
    critic_out = agent.critic(dummy_obs, dummy_act)
    print(f"  Actor  output shape : {tuple(actor_out.shape)}   "
          f"(expected ({BATCH}, {ACT_DIM}))")
    print(f"  Critic output shape : {tuple(critic_out.shape)}  "
          f"(expected ({BATCH},))")
    assert actor_out.shape  == (BATCH, ACT_DIM), "Actor shape mismatch"
    assert critic_out.shape == (BATCH,),         "Critic shape mismatch"

    # ── [2] Actor output is in (−1, +1) ──────────────────────────────
    print("\n[2] Actor Tanh output bounds")
    assert actor_out.abs().max().item() <= 1.0, "Actor output outside (−1,+1)"
    print(f"  max |actor output| = {actor_out.abs().max().item():.4f}  ≤ 1 ✓")

    # ── [3] Target networks are frozen ───────────────────────────────
    print("\n[3] Target network requires_grad=False")
    for name, param in agent.actor_targ.named_parameters():
        assert not param.requires_grad, f"{name} has requires_grad=True"
    for name, param in agent.critic_targ.named_parameters():
        assert not param.requires_grad, f"{name} has requires_grad=True"
    print("  All target parameters frozen ✓")

    # ── [4] Replay buffer store and sample ───────────────────────────
    print("\n[4] Replay buffer")
    obs_np = np.zeros(OBS_DIM, dtype=np.float32)
    for i in range(CAPACITY + 10):   # overfill to test circular wrap
        buffer.store(obs_np, np.zeros(ACT_DIM), 0.0, obs_np, False)
    assert len(buffer) == CAPACITY,  "Buffer size should be capped at capacity"
    assert buffer.ptr  == 10,        "Write pointer should have wrapped"
    batch = buffer.sample(BATCH, device)
    assert batch['obs'].shape  == (BATCH, OBS_DIM), "Sampled obs shape wrong"
    assert batch['act'].shape  == (BATCH, ACT_DIM), "Sampled act shape wrong"
    assert batch['rew'].shape  == (BATCH,),          "Sampled rew shape wrong"
    print(f"  Buffer filled to {len(buffer)} (capacity {CAPACITY}), "
          f"ptr wrapped to {buffer.ptr} ✓")
    print(f"  Sampled batch shapes: obs={tuple(batch['obs'].shape)}, "
          f"act={tuple(batch['act'].shape)}, rew={tuple(batch['rew'].shape)} ✓")

    # ── [5] Full update step — check losses are finite scalars ────────
    print("\n[5] Full agent.update() call")
    # Fill buffer with random transitions from the environment
    buffer2 = ReplayBuffer(OBS_DIM, ACT_DIM, capacity=CAPACITY)
    obs_e, _ = env.reset(seed=0)
    for _ in range(CAPACITY):
        act_e = env.action_space.sample()
        obs_e2, rew_e, term_e, trunc_e, _ = env.step(act_e)
        buffer2.store(obs_e, act_e, rew_e, obs_e2, term_e)
        obs_e = obs_e2
        if term_e or trunc_e:
            obs_e, _ = env.reset()

    batch2 = buffer2.sample(BATCH, device)
    c_loss, a_loss = agent.update(batch2)
    assert np.isfinite(c_loss), f"Critic loss is not finite: {c_loss}"
    assert np.isfinite(a_loss), f"Actor  loss is not finite: {a_loss}"
    print(f"  critic_loss = {c_loss:.6f}  actor_loss = {a_loss:.6f} ✓")

    # ── [6] Polyak update moved targets ──────────────────────────────
    print("\n[6] Polyak update correctness")
    # After one update the actor target must differ from the actor main
    # (they were equal at init; the main has changed, target partially follows)
    main_params  = [p.data.clone() for p in agent.actor.parameters()]
    targ_params  = [p.data.clone() for p in agent.actor_targ.parameters()]
    diff = sum((m - t).abs().sum().item()
               for m, t in zip(main_params, targ_params))
    print(f"  |main − target| param diff after update: {diff:.6f}  "
          f"(> 0 expected) ✓")
    assert diff > 0, "Target should have diverged from main after update"

    # ── [7] select_action with and without noise ──────────────────────
    print("\n[7] select_action")
    obs_e, _ = env.reset()
    act_noisy = agent.select_action(obs_e, add_noise=True)
    act_det   = agent.select_action(obs_e, add_noise=False)
    assert act_noisy.shape == (ACT_DIM,), "Noisy action shape wrong"
    assert act_det.shape   == (ACT_DIM,), "Det action shape wrong"
    assert np.all(np.abs(act_noisy) <= 1.0), "Noisy action out of [−1,+1]"
    assert np.all(np.abs(act_det)   <= 1.0), "Det action out of [−1,+1]"
    print(f"  det_action  = {act_det}  (no noise) ✓")
    print(f"  noisy_action= {act_noisy} (with noise) ✓")

    # ── [8] evaluate() runs without error ────────────────────────────
    print("\n[8] evaluate()")
    stats = evaluate(agent, env, n_episodes=3)
    for k, v in stats.items():
        assert np.isfinite(v), f"{k} is not finite"
    print(f"  mean_return={stats['mean_return']:.3f}  "
          f"mean_length={stats['mean_length']:.1f}  "
          f"success_rate={stats['success_rate']:.3f} ✓")

    print("\n" + "=" * 60)
    print("  All smoke tests passed ✓")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    run_smoke_test()

    # ── EmptyGridEnv demo run ─────────────────────────────────────────────────
    # 20 000 steps on the open grid.  Expect EvSucc > 0 here: if the algorithm
    # works at all, an obstacle-free environment should show early learning.
    print("\n" + "=" * 75)
    print("  Demo training run — EmptyGridEnv  (20 000 steps)")
    print("  Goal: confirm DDPG learns on a trivial layout before the maze.")
    print("=" * 75 + "\n")

    #train(
    #    env_class        = EmptyGridEnv,
    #    epochs           = 10,
    #    steps_per_epoch  = 2_000,
    #    start_steps      = 2_000,
    #    update_after     = 500,
    #    update_every     = 1,
    #    batch_size       = 256,
    #    gamma            = 0.98,
    #    polyak           = 0.995,
    #    act_noise        = 0.1,
    #    n_eval_episodes  = 10,
    #    save_freq        = 5,
    #    seed             = 42,
    #    checkpoint_dir   = 'checkpoints_empty',
    #)

    # ── Full runs (run locally, preferably with GPU) ──────────────────────────
    # EmptyGridEnv full run:
    #train(env_class=EmptyGridEnv,     epochs=100, steps_per_epoch=4_000,
    #       start_steps=5_000, update_after=1_000, seed=42,
    #       checkpoint_dir='checkpoints_empty_full',
    #       results_dir='results_empty_full')
    #
    # FourRoomMazeEnv full run:
    train(env_class=FourRoomMazeEnv,  epochs=100, steps_per_epoch=4_000,
           start_steps=5_000, update_after=1_000, seed=42,
           checkpoint_dir='checkpoints_maze_full')
