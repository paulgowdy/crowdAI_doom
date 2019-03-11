
parallel_size = 2

use_pixel_change = True
use_value_replay = True
use_reward_prediction = True

pixel_change_lambda = 0.05 # 0.05, 0.01 ~ 0.1 for lab
entropy_beta = 0.001 # "entropy regularization constant"

initial_alpha_low = 1e-4
initial_alpha_high = 5e-3
initial_alpha_log_rate = 0.5

rmsp_alpha = 0.99 # "decay parameter for rmsprop"
rmsp_epsilon = 0.1 # "epsilon parameter for rmsprop"
grad_norm_clip = 40.0 # "gradient norm clipping"

env_type = "doom"
env_name = "wad_6"

local_t_max = 20
gamma = 0.99
gamma_pc = 0.9
experience_history_size = 2000
max_time_step = 10 * 10**7

log_file = "/tmp/unreal_log/unreal_log" #, "log file directory"

save_interval_step = 1000 * 100 # 100 * 1000 # "saving interval steps"
checkpoint_dir = "/tmp/unreal_checkpoints" # "checkpoint directory"

'''
tf.app.flags.DEFINE_boolean("use_pixel_change", True, "whether to use pixel change")
  tf.app.flags.DEFINE_boolean("use_value_replay", True, "whether to use value function replay")
  tf.app.flags.DEFINE_boolean("use_reward_prediction", True, "whether to use reward prediction")

  tf.app.flags.DEFINE_string("checkpoint_dir", "/tmp/unreal_checkpoints", "checkpoint directory")

  # For training
  if option_type == 'training':
    tf.app.flags.DEFINE_integer("parallel_size", 8, "parallel thread size")
    tf.app.flags.DEFINE_integer("local_t_max", 20, "repeat step size")
    tf.app.flags.DEFINE_float("rmsp_alpha", 0.99, "decay parameter for rmsprop")
    tf.app.flags.DEFINE_float("rmsp_epsilon", 0.1, "epsilon parameter for rmsprop")

    tf.app.flags.DEFINE_string("log_file", "/tmp/unreal_log/unreal_log", "log file directory")
    tf.app.flags.DEFINE_float("initial_alpha_low", 1e-4, "log_uniform low limit for learning rate")
    tf.app.flags.DEFINE_float("initial_alpha_high", 5e-3, "log_uniform high limit for learning rate")
    tf.app.flags.DEFINE_float("initial_alpha_log_rate", 0.5, "log_uniform interpolate rate for learning rate")
    tf.app.flags.DEFINE_float("gamma", 0.99, "discount factor for rewards")
    tf.app.flags.DEFINE_float("gamma_pc", 0.9, "discount factor for pixel control")
    tf.app.flags.DEFINE_float("entropy_beta", 0.001, "entropy regularization constant")
    tf.app.flags.DEFINE_float("pixel_change_lambda", 0.05, "pixel change lambda") # 0.05, 0.01 ~ 0.1 for lab, 0.0001 ~ 0.01 for gym
    tf.app.flags.DEFINE_integer("experience_history_size", 2000, "experience replay buffer size")
    tf.app.flags.DEFINE_integer("max_time_step", 10 * 10**7, "max time steps")
    tf.app.flags.DEFINE_integer("save_interval_step", 100 * 1000, "saving interval steps")
    tf.app.flags.DEFINE_boolean("grad_norm_clip", 40.0, "gradient norm clipping")


'''
