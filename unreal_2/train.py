import math
import numpy as np
import tensorflow as tf

from environment import *
from model import UnrealModel
from rmsprop_applier import RMSPropApplier
from trainer import Trainer

import flags

USE_GPU = False



#############################################

def log_uniform(lo, hi, rate):
  log_lo = math.log(lo)
  log_hi = math.log(hi)
  v = log_lo * (1-rate) + log_hi * rate
  return math.exp(v)

'''
def signal_handler(self, signal, frame):
	print('You pressed Ctrl+C!')
	terminate_reqested = True
'''

def train_function(parallel_index, preparing):

def save():


device = "/cpu:0"

if USE_GPU:
  device = "/gpu:0"

initial_learning_rate = log_uniform(initial_alpha_low,
                                        initial_alpha_high,
                                        initial_alpha_log_rate)
global_t = 0

action_size = len(action_list)


global_network = UnrealModel(action_size,
                                      -1,
                                      flags.use_pixel_change,
                                      flags.use_value_replay,
                                      flags.use_reward_prediction,
                                      flags.pixel_change_lambda,
                                      flags.entropy_beta,
                                      device)



learning_rate_input = tf.placeholder("float")

grad_applier = RMSPropApplier(learning_rate = learning_rate_input,
                                  decay = flags.rmsp_alpha,
                                  momentum = 0.0,
                                  epsilon = flags.rmsp_epsilon,
                                  clip_norm = flags.grad_norm_clip,
                                  device = device)

trainers = []

for i in range(flags.parallel_size):

	print('creating trainer', i)
	trainer = Trainer(i,
	                global_network,
	                initial_learning_rate,
	                learning_rate_input,
	                grad_applier,
	                flags.env_type,
	                flags.env_name,
	                flags.use_pixel_change,
	                flags.use_value_replay,
	                flags.use_reward_prediction,
	                flags.pixel_change_lambda,
	                flags.entropy_beta,
	                flags.local_t_max,
	                flags.gamma,
	                flags.gamma_pc,
	                flags.experience_history_size,
	                flags.max_time_step,
	                device)

	trainers.append(trainer)
	print('')

config = tf.ConfigProto(log_device_placement=False,allow_soft_placement=True)
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

# summary for tensorboard
score_input = tf.placeholder(tf.int32)
tf.summary.scalar("score", score_input)

summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(flags.log_file, sess.graph)

# init or load checkpoint with saver
saver = tf.train.Saver(global_network.get_vars())

checkpoint = tf.train.get_checkpoint_state(flags.checkpoint_dir)

if checkpoint and checkpoint.model_checkpoint_path:
	self.saver.restore(sess, checkpoint.model_checkpoint_path)
	print("checkpoint loaded:", checkpoint.model_checkpoint_path)
	tokens = checkpoint.model_checkpoint_path.split("-")
	# set global step
	global_t = int(tokens[1])
	print(">>> global step set: ", global_t)
	# set wall time
	wall_t_fname = flags.checkpoint_dir + '/' + 'wall_t.' + str(global_t)

	with open(wall_t_fname, 'r') as f:
		wall_t = float(f.read())
		next_save_steps = (global_t + flags.save_interval_step) // flags.save_interval_step * flags.save_interval_step

else:
	print("Could not find old checkpoint")
	# set wall time
	wall_t = 0.0
	next_save_steps = flags.save_interval_step


# run training threads
train_threads = []

for i in range(flags.parallel_size):

	train_threads.append(threading.Thread(target=train_function, args=(i,True)))

#signal.signal(signal.SIGINT, signal_handler)

# set start time
start_time = time.time() - wall_t

for t in train_threads:
	t.start()

#print('Press Ctrl+C to stop')
#signal.pause()
