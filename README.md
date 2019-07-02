# RIPPE: Robotic Interactive Physics Parameters Estimator

This repository holds the code used for our recent work on estimating physical parameters from robot interactions.
Our experiments show how a robot can have a sense of consistent physical parameters even when only sparse noisy data is available.

## Instructions for running the code
I usually build pybullet from sources.
In this case, you need the following information in your `$PYTHONPATH` variable:

```
$ export PYTHONPATH=$PYTHONPATH:$BULLET/build_cmake/examples/pybullet:$BULLET/examples/pybullet/gym
```

Assuming `$BULLET` is the directory where you cloned bullet repository.

In this case, you should be able to run:

```
$ python rippe_parallel.py
```

Which relies on `joblib` to parallelize the simulation execution and you can change the parameters in `parametersConfig.py`.

Warning: The file `rippe_nonparallel.py` was used for debugging and ignores the information in `parametersConfig.py`.
You have to manually change the source for your desired parameters.

## Citation
If you use any part of this program, we kindly ask you to cite the following related publication:
```
@inproceedings{dehban2019robotic,
  title={Robotic Interactive Physics Parameters Estimator (RIPPE)},
  author={Dehban, Atabak and Cardoso, Carlos and  Vicente, Pedro and Bernardino, Alexandre and Santos-Victor, Jos{\'e}},
  booktitle="IEEE International Conference on Developmental and Learning and on Epigenetic Robotics (ICDL-Epirob)",
  year={2019},
  note={in press}
}
```
  