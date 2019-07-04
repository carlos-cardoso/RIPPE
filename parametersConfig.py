
dbounds = {'lateralFriction': (0.01, 5.0),
           'spinningFriction': (1.0e-4, 0.5),
           'rollingFriction': (1.0e-12, 1.0e-3),
           'restitution': (0.0001, 0.95),
           'mass': (0.0001, 0.2),
           'xnoise': (0.0, 0.05),
           'ynoise': (0.0, 0.05),
           'xmean': (-0.02, 0.02),
           'ymean': (-0.02, 0.02)}

# param_names = ['lateralFriction', 'spinningFriction', 'rollingFriction',]# 'restitution']
param_names = ['lateralFriction', 'rollingFriction', 'mass']# 'restitution']
# param_names = ['mass', ]# 'restitution']

train_tools = ("rake",)
train_actions = ("tap_from_left", "push")
object_name = "lemon"
test_tools = ("hook",)
test_actions = ("tap_from_right",)


N_EXPERIMENTS = 50  # running experiment per object per tool
N_TRIALS = 50  # optimization steps
from skopt import gp_minimize, forest_minimize, dummy_minimize
optimizer = gp_minimize