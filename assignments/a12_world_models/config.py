"""Hyperparameters for the DreamerV3 cartpole-from-pixels build. Provided, not a hole.

The values are those of r2dreamer (github.com/NM512/r2dreamer, commit becf338) in its DreamerV3
mode (`model.rep_loss=dreamer`), with the `size12M` model and the `dmc_vision` environment config.
Runs use `env_num=4, eval_episode_num=2, buffer.max_size=1e5`, which are the defaults below. The
DreamerV3 paper is arXiv 2301.04104v2.

Measured on one RTX 4080 laptop GPU with torch.compile:
- cartpole-swingup (the default task): one run of this port reached eval return 856 at 200k
  environment steps and 864-876 from 220k to 290k (about 9.5 min per 10k steps at full GPU clock).
  Uniform-random actions score about 43 (50 episodes) and zero force 0.
- cartpole-balance (`--task balance`): the reference reached 991, 984 and 969 (seeds 0, 1, 2; mean
  of the evals at 70k, 80k, 90k steps) and this port 966, 920 and 975. Uniform random scores about
  310 and zero force about 750.

Field names follow the paper where it has a symbol and r2dreamer otherwise; the reference config
key is given in a comment when it differs.
"""

from dataclasses import dataclass, field


@dataclass
class Config:
    # ---- environment (env/dmc_vision.yaml) ----
    obs_size: int = 64              # square RGB render, stored as uint8 (3, 64, 64)
    action_dim: int = 1             # cartpole: one force in [-1, 1]
    action_repeat: int = 2          # each agent step applies the force for 2 physics steps
    time_limit: int = 1000          # physics steps per episode, so 500 agent steps

    # ---- RSSM (model/_base_.yaml rssm, model/size12M.yaml) ----
    deter: int = 2048               # width of the deterministic state h
    hidden: int = 256               # width of the RSSM input projections and logit MLPs
    stoch: int = 32                 # number of categorical variables in z
    discrete: int = 16              # classes per categorical
    blocks: int = 8                 # block-diagonal GRU: 8 independent blocks of 256 units
    dyn_layers: int = 1             # block-linear layers before the GRU gates
    obs_layers: int = 1             # hidden layers of the posterior logit MLP
    img_layers: int = 2             # hidden layers of the prior logit MLP
    unimix: float = 0.01            # uniform mixture in every categorical of z

    # ---- encoder / decoder CNN (model/_base_.yaml encoder.cnn, decoder.cnn) ----
    cnn_depth: int = 16
    cnn_mults: tuple = (2, 3, 4, 4)  # channels 32, 48, 64, 64; 64x64 -> 4x4
    kernel_size: int = 5
    dec_bspace: int = 8             # blocks of the decoder's block-linear projection of h
    units: int = 256                # MLP width (heads, actor, critic, decoder z projection)

    # ---- heads ----
    actor_layers: int = 3
    critic_layers: int = 3
    reward_layers: int = 1
    cont_layers: int = 1
    bins: int = 255                 # two-hot bins for reward and critic
    actor_min_std: float = 0.1
    actor_max_std: float = 1.0
    actor_outscale: float = 0.01    # last-layer weights scaled down at init
    reward_outscale: float = 0.0    # zero init: the head predicts 0 until trained
    critic_outscale: float = 0.0
    cont_outscale: float = 1.0

    # ---- losses (model/_base_.yaml loss_scales, kl_free) ----
    kl_free: float = 1.0            # free bits, nats, on the KL summed over the 32 categoricals
    loss_scales: dict = field(default_factory=lambda: {
        "image": 1.0,               # `recon`
        "rew": 1.0,
        "con": 1.0,
        "dyn": 1.0,                 # beta_dyn
        "rep": 0.1,                 # beta_rep
        "policy": 1.0,
        "value": 1.0,
        "repval": 0.3,              # critic loss on replayed posterior states
    })

    # ---- actor-critic ----
    imag_horizon: int = 15          # imagined steps; 16 states including the start
    horizon: int = 333              # discount = 1 - 1/333, about 0.997
    lamb: float = 0.95              # lambda of the lambda-return
    act_entropy: float = 3e-4       # entropy bonus eta
    ret_ema_alpha: float = 0.01     # EMA rate of the 5th/95th return percentiles
    ret_quantiles: tuple = (0.05, 0.95)
    slow_target_update: int = 1     # update the slow critic every N updates
    slow_target_fraction: float = 0.02

    # ---- optimizer (LaProp + adaptive gradient clipping, one optimizer for everything) ----
    lr: float = 4e-5
    agc: float = 0.3
    pmin: float = 1e-3
    eps: float = 1e-20
    beta1: float = 0.9
    beta2: float = 0.999
    warmup: int = 1000              # linear LR warmup, in updates

    # ---- replay and schedule (configs.yaml) ----
    batch_size: int = 16
    batch_length: int = 64
    train_ratio: int = 512          # replayed steps per environment step (counting action repeat)
    replay_size: int = 100_000      # total transitions over all envs (`buffer.max_size`)
    online_queue: bool = True       # train on freshly collected windows first (`buffer.online`)
    pretrain: int = 0               # updates on the first training call (`trainer.pretrain`)
    env_num: int = 4                # parallel training envs
    eval_episodes: int = 2          # eval envs, one episode each per evaluation
    eval_every: int = 10_000        # environment steps
    steps: int = 200_000            # environment steps (counting action repeat)
    log_every: int = 5_000          # environment steps between training-metric lines

    # ---- runtime ----
    seed: int = 0
    task: str = "swingup"            # dm_control cartpole task: "swingup" or "balance"
    device: str = "cuda"
    compute_dtype: str = "bfloat16"  # autocast dtype; "float32" disables autocast
    compile: bool = False            # torch.compile the update step; the CLI turns it on for CUDA


def small_config(**overrides) -> Config:
    """A shrunk config for CPU smoke tests: same structure, much smaller widths."""
    base = dict(deter=64, hidden=32, stoch=4, discrete=4, blocks=2, cnn_depth=4, units=32,
                bins=31, batch_size=2, batch_length=8, train_ratio=16, device="cpu",
                compute_dtype="float32")
    base.update(overrides)
    return Config(**base)
