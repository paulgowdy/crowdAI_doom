import numpy as np

class Environment(object):
  # cached action size
  action_size = -1

  @staticmethod
  def create_environment(env_type, env_name):
    #from doom_environment import DoomEnvironment
    import doom_environment
    print('returning doom env')
    return doom_environment.DoomEnvironment('doom')

  @staticmethod
  def get_action_size(env_type, env_name):
    return 4

  def __init__(self):
    pass

  def process(self, action):
    pass

  def reset(self):
    pass

  def stop(self):
    pass

  def _subsample(self, a, average_width):
    s = a.shape
    sh = s[0]//average_width, average_width, s[1]//average_width, average_width
    return a.reshape(sh).mean(-1).mean(1)

  def _calc_pixel_change(self, state, last_state):
    d = np.absolute(state[2:-2,2:-2,:] - last_state[2:-2,2:-2,:])
    # (80,80,3)
    m = np.mean(d, 2)
    c = self._subsample(m, 4)
    return c
