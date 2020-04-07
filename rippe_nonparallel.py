#!/usr/bin/python3
from __future__ import print_function, division, absolute_import

import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import stl
from pprint import pprint
import pybullet as p
import pybullet_data
import signal
from skopt import gp_minimize, dump, load, forest_minimize, dummy_minimize
from tqdm import tqdm, tqdm_notebook
import os
import sys
from contextlib import contextmanager
from matplotlib import pyplot as plt
#from universal_divergence import estimate as KLD
from continuous_kl import KL_from_distributions as KLD
import logging
import time



N_EXPERIMENTS = 100  # running experiment per object per tool
N_TRIALS = 50  # optimization steps
PYBULLET_INSTANCE = p.GUI
WATCHDOG = True
PLOTTING = True

logging.basicConfig(filename="log",
                      filemode='a',
                      format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                      datefmt='%H:%M:%S',
                      level=logging.DEBUG)



@contextmanager
def stdout_redirected(to=os.devnull):
  '''
  import os

  with stdout_redirected(to=filename):
      print("from Python")
      os.system("echo non-Python applications are also supported")
  '''
  fd = sys.stdout.fileno()

  ##### assert that Python and C stdio write using the same file descriptor
  ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

  def _redirect_stdout(to):
    # sys.stdout.close() # + implicit flush()
    os.dup2(to.fileno(), fd)  # fd writes to 'to' file
    # sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

  with os.fdopen(os.dup(fd), 'w') as old_stdout:
    with open(to, 'w') as file:
      _redirect_stdout(to=file)
    try:
      yield  # allow code to be run with the redirected stdout
    finally:
      _redirect_stdout(to=old_stdout)  # restore stdout.
      # buffering and flags such as
      # CLOEXEC may be different


def handler(signum, frame):
  print("FROZEN")
  raise ValueError


"""
model	weight(g)	object	description
14	29	lemon	lemon
16	49	pear	pear
17	47	orange	orange
56	58	yball	tennis
57	41	bball	racquet
58	46	wball	golf
65G	28	ocup	medium orange cup
65J	38	ycup	big yellow cup
65D	19	sycup	small yellow cup
73E	26	elego	Lego bridge
73D	16	dlego	Lego eye
"""

models = {
  "lemon": ["14", 29.0],  # lemon
  "pear": ["16", 49.0],  # pear
  "orange": ["17", 47.0],  # orange
  "yball": ["56", 58.0],  # tennis
  "bball": ["57", 41.0],  # racquet
  "wball": ["58", 46.0],  # golf
  "ocup": ["65G", 28.0],  # medium orange cup
  "ycup": ["65J", 38.0],  # big yellow cup
  "sycup": ["65D", 19.0],  # small yellow cup
  "elego": ["73E", 26.0],  # Lego bridge
  "ylego": ["73D", 16.0],  # Lego eye
}


def load_experiment(fname="effData.txt", get_eff_data=False):
  colnames = ['toolName', 'targetName', 'actionId', 'initialObjPos[0]', 'initialObjPos[1]', 'initialObjImgPos.x',
              'initialObjImgPos.y', 'finalObjectPos[0]',
              'finalObjectPos[1]', 'finalObjImgPos.x', 'finalObjImgPos.y']
  df = pd.read_csv(fname, delim_whitespace=True, names=colnames)

  diffx = (df['finalObjectPos[0]'] - df['initialObjPos[0]'])
  diffy = (df['finalObjectPos[1]'] - df['initialObjPos[1]'])

  mvec = np.array([diffx.mean(), diffy.mean()])
  mdist = np.mean(np.sqrt(diffx ** 2 + diffy ** 2))
  vvec = np.array([diffx.var(), diffy.var()])

  weight = 1e-3 * models[df['targetName'][0]][1]

  return (mvec, vvec, weight, mdist, np.vstack([diffx, diffy]).T) if get_eff_data else (mvec, vvec, weight, mdist)


def call_simulator():
  # print("start sim")
  physicsClient = p.connect(PYBULLET_INSTANCE)  # or p.DIRECT for non-graphical version

  p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
  p.setGravity(0, 0, -9.8)
  p.setPhysicsEngineParameter(enableFileCaching=0)#, fixedTimeStep=0.0001)
  planeId = p.loadURDF("plane.urdf")


def reset_world():
  p.resetSimulation()


def load_object():
    boxId = p.loadURDF("models/object.urdf")
    return boxId


def load_robot(toolname):
  print(toolname)
  toolId = p.loadSDF("models/{}.sdf".format(toolname))
  return toolId


def delete_object(objID):
  p.removeBody(objID)


def delete_robot(robotID):
  p.removeBody(robotID)


def get_obj_xy(objID):
  pos, ori = p.getBasePositionAndOrientation(objID)
  pos = pos[0:-1]
  return pos


def gen_object(weight=0.3, mu1=0.3, mu2=0.3, slip=0.0, iner=np.eye(3), center_of_mass=np.zeros((3,)),
               object_name="object", startpose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)):
  tree = ET.parse('models/template_object.urdf')
  root = tree.getroot()

  world = root[0]
  for mesh in root.findall(".//mesh"):
    mesh.attrib['filename'] = "models/cvx_{}.stl".format(object_name)

  # Set model mass
  for mass in root.findall(".//mass"):
    mass.attrib['value'] = str(weight)

  for origin in root.findall(".//origin"):
    origin.attrib["xyz"] = str(center_of_mass)[1:-1]

  for inert in root.findall(".//inertia"):
    inert.attrib['ixx'] = str(iner[0, 0])
    inert.attrib['ixy'] = str(iner[0, 1])
    inert.attrib['ixz'] = str(iner[0, 2])
    inert.attrib['iyy'] = str(iner[1, 1])
    inert.attrib['iyz'] = str(iner[1, 2])
    inert.attrib['izz'] = str(iner[2, 2])

  tree.write('models/object.urdf')


def gen_robot(action_name, tool_name):
  tree = ET.parse('models/template_robot_{}.sdf'.format(action_name))
  root = tree.getroot()

  # Set robot pose
  for model in root.findall(".//model[@name='robot']"):
    for uri in model.findall(".//uri"):
      uri.text = "models/cvx_{}.obj".format(tool_name)


  tree.write('models/robot.sdf')


def cost(x,y):
  return (100*(np.linalg.norm(x.mean(0)-y.mean(0)) + np.linalg.norm(x.var(0)-y.var(0))))**2


def gen_run_experiment(pbar, param_names,object_name="yball", tools=("rake","stick", "hook"),
                       actions=("tap_from_left", "push", "draw", "tap_from_right"),
                       # actions=("tap_from_left", "tap_from_right"),
                       # actions=("tap_from_right",),
                       ):
  # get properties:
  object_mesh = stl.Mesh.from_file("models/cvx_{}.stl".format(object_name))
  props = object_mesh.get_mass_properties()
  center_of_mass = props[1]
  inertia_tensor = props[2]
  gen_object(weight=0.1, mu1=0.1, mu2=0.1, slip=0.1, iner=inertia_tensor,
             center_of_mass=center_of_mass, object_name=object_name, startpose=(0, 0, 0.05, 0, 0, 0))
  objID = load_object()


  def f(params):
    dic_params = {pname: param for pname, param in zip(param_names, params)}
    offset = 0.01 if object_name == "yball" else 0.0
    init_poses = {"rake": {"push": np.array([0.0, 0.0, 0.0, -0.04 - offset, 0.0]),
                           "draw": np.array([0.0, 0.0, 0.0, 0.055 + offset, 0.025]),
                           "tap_from_right": np.array([0.0, 0.0, 0.0, 0.03, -0.145 - offset]),
                           "tap_from_left": np.array([0.0, 0.0, 0.0, 0.02, 0.14 + offset])
                           },
                  "stick": {"push": np.array([0.0, 0.0, 0.0, -0.035 - offset, 0.0]),
                            "draw": np.array([0.0, 0.0, 0.0, 0.035 + offset, 0.03]),
                            "tap_from_right": np.array([0.0, 0.0, 0.0, 0.06, -0.065 - offset]),
                            "tap_from_left": np.array([0.0, 0.0, 0.0, 0.06, 0.05 + offset])
                            },
                  "hook": {"push": np.array([0.0, 0.0, 0.0, -0.04 - offset, 0.0]),
                           "draw": np.array([0.0, 0.0, 0.0, 0.15 + offset, 0.05]),
                           "tap_from_right": np.array([0.0, 0.0, 0.0, 0.06, -0.075 - offset]),
                           "tap_from_left": np.array([0.0, 0.0, 0.0, 0.09, 0.10 + offset])
                           }

                  }


    costs = list()
    sim_eff_history = np.zeros((N_EXPERIMENTS, 2))
    for tool_name in tools:
      for action_name in actions:
        target_pos, target_var, gnd_weight, mdist, real_eff_history = load_experiment(
                  "affordance-datasets/visual-affordances-of-objects-and-tools/{}/{}/{}/effData.txt".format(tool_name, object_name, action_name), get_eff_data=True)
        for iter in range(N_EXPERIMENTS):
          success = False
          while not success:
            try:
              if WATCHDOG:
                signal.alarm(10)

              with stdout_redirected():
                robotID = load_robot(tool_name)
                toolID = robotID[0]

              nonlocal objID
              if (toolID == objID):
                raise ValueError


              mu = init_poses[tool_name][action_name]
              yaw, pitch, roll, x, y = np.random.normal(mu, np.array([1.0, 1.0, 1.0, 0.0, 0.0]))
              initial_xy = np.array([x, y])

              p.resetBasePositionAndOrientation(objID, posObj=[x, y, 0.05], ornObj=[yaw, pitch, roll, 1])
              dic_params2=dict(dic_params)
              dic_params2['linearDamping']=0.0
              dic_params2['angularDamping']=0.0
              #dic_params2['mass']=0.001*models[object_name][1]
              #dic_params2['rollingFriction']=0.00001
              #dic_params['lateralFriction']=0.0001
              #dic_params['spinningFriction']=0.0001
              #dic_params['rollingFriction']=0.0001
              #del dic_params2['xnoise']
              #del dic_params2['ynoise']
              p.changeDynamics(objID, -1, **dic_params2)

              get_obj_xy(objID)

              speed = 0.04

              if action_name == "push":
                base_speed = [-speed, 0, 0]
                base_pos_limit = lambda js: js[0] <= -0.12
              elif action_name == "draw":
                base_speed = [speed, 0, 0]
                base_pos_limit = lambda js: js[0] >= 0.12
              elif action_name == "tap_from_left":
                base_speed = [0, speed, 0]
                base_pos_limit = lambda js: js[1] >= 0.12
              elif action_name == "tap_from_right":
                base_speed = [0, -speed, 0]
                base_pos_limit = lambda js: js[1] <= -0.12
              else:
                raise ValueError

              # Let object fall to the ground and stop it
              pxyz, pori = p.getBasePositionAndOrientation(objID)
              position_after_fall = 100 * np.ones_like(pxyz)
              orientation_after_fall = 100 * np.ones_like(pori)
              while not np.allclose(position_after_fall[-1:], pxyz[-1:], atol=1e-6):
                p.stepSimulation()
                pxyz = position_after_fall
                position_after_fall, orientation_after_fall = p.getBasePositionAndOrientation(objID)



              p.resetBasePositionAndOrientation(objID, posObj=[initial_xy[0], initial_xy[1], position_after_fall[2]] , ornObj=orientation_after_fall)
              p.resetBaseVelocity(objID, [0.0, 0.0, 0.0])
              ppos = get_obj_xy(objID)
              npos = 100 * np.ones_like(ppos)
              iters = 0

              # Move tool
              p.resetBaseVelocity(toolID, base_speed)
              action_finnished = False
              while not np.allclose(npos, ppos, atol=1e-6) or iters < 100:
                js, jor = p.getBasePositionAndOrientation(toolID)

                if (base_pos_limit(js)):
                  p.resetBaseVelocity(toolID, [0, 0, 0])
                  action_finnished = True
                elif not action_finnished:
                  p.resetBaseVelocity(toolID, base_speed)

                p.stepSimulation()
                ppos = npos
                npos = get_obj_xy(objID)
                if action_finnished:
                  iters += 1
                # time.sleep(1./1000.)

              pos = npos

              delete_robot(toolID)
              sim_eff_history[iter] = pos - initial_xy

              success = True
            except ValueError:
              p.resetSimulation()
              p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
              p.setGravity(0, 0, -9.8)
              p.setPhysicsEngineParameter(enableFileCaching=0)
              planeId = p.loadURDF("plane.urdf")
              p.changeDynamics(planeId, -1, restitution=.97, linearDamping=0.0, angularDamping=0.0)
              objID = load_object()



        kld = KLD(real_eff_history, sim_eff_history)
        if PLOTTING:
          plt.scatter(real_eff_history[:,1], -real_eff_history[:,0], s=40,c="red", edgecolors='none', label="real")
          plt.scatter(sim_eff_history[:, 1], -sim_eff_history[:, 0], s=40, c="blue", edgecolors='none', label="sim")
          plt.legend(loc=2)
          plt.ylim(-0.3, 0.3)
          plt.xlim(-0.3, 0.3)
          plt.title('action: {}, tool: {}'.format(action_name, tool_name))
          plt.xlabel('[m]')
          plt.ylabel('[m]')
          plt.savefig("{}.png".format(kld))
          plt.savefig("{}.pdf".format(kld))
          plt.show()
        costs.append(kld)
        sim_eff_history = np.zeros((N_EXPERIMENTS, 2))
        logging.info("cost:{}".format(kld))

    if WATCHDOG:
      signal.alarm(0)
    out = sum(costs)/len(costs)
    print('\033[93m' + str(dic_params)+'\033[0m')
    pbar.set_description('cost: %0.2f' % (out))
    pbar.update(1)
    return out

  return f


def optimize(param_names, fname, object_name="yball", opt_f=gp_minimize):
  dbounds = {'lateralFriction': (0.01, 5.0), 'spinningFriction': (1.0e-4, 5.0), 'rollingFriction': (1.0e-12, 1.0e-3),
             'restitution': (0.0001, 0.95), 'mass': (0.0001, 10.0), 'xnoise': (0.0, 0.05), 'ynoise' : (0.0, 0.05),
             'xmean': (-0.02, 0.02), 'ymean': (-0.02, 0.02)}
  pbounds = [dbounds[param] for param in param_names]

  with tqdm(total=N_TRIALS - 1, file=sys.stdout) as pbar:
    run_experiment = gen_run_experiment(pbar, param_names, object_name=object_name)
    res = opt_f(run_experiment, pbounds, n_calls=N_TRIALS)
  dump(res, fname, store_objective=False)
  return res


def test(param_names, fname, obj_name):
  test_tools = ("hook",)
  test_actions = ("tap_from_right",)

  with tqdm(total=N_TRIALS - 1, file=sys.stdout) as pbar:
    func = gen_run_experiment(pbar, param_names, tools=test_tools, actions=test_actions, object_name=obj_name)

    c_all = []
    costs = []
    best = []
    best_iters = []
    best_params = []
    max_target = 1000.0
    res = load(fname)

    for ind, xi in enumerate(res.x_iters):
      pprint(res.func_vals[ind])
      ctarget = res.func_vals[ind]
      c_all.append(ctarget)
      if ctarget < max_target:
        max_target = ctarget
        best.append(ctarget)
        best_iters.append(ind)
        print("new best:{}".format(ctarget))
        params = xi  # data['params']
        best_params.append(params)
        pprint(params)
        c = func(params)
        costs.append(c)


  print(c_all)
  logging.info("tr costs:{}".format(c_all))
  print(best)
  logging.info("tr best:{}".format(best))
  print(costs)
  logging.info("tst best:{}".format(costs))
  print(best_params)
  logging.info("best params:{}".format(best_params))
  print(ind)
  logging.info("ind:{}".format(ind))


if __name__ == "__main__":

  logging.info("RIPPE")

  if WATCHDOG:
    signal.signal(signal.SIGALRM, handler)

  call_simulator()
  param_names = ['lateralFriction', 'rollingFriction', 'mass']#, 'xnoise', 'ynoise']

  object_name = ["ylego",]
  # object_name = ["lemon",]

  randn=np.random.random(1)[0]

  for obj in object_name:
    fname = "saved/gp_fixed_{}.bz2".format(obj)
    logging.info("file name: "+ fname)
    optimize(param_names, fname, obj, opt_f=gp_minimize)
    test(param_names, fname, obj)
