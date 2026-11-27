# GaBP Tutorial

This project adds range and range-bearing factors and a sliding-window `MultiRobotGraph` to gsolver. The C++ libraries use `SE3PoseVel` and support gsolver's built-in pose prior and odometry factors. The `gabp` Python module exposes these components for use with gsolver's Python types. Both the gsolver and GTSAM examples use the UTIAS MRCLAM Dataset 1 included under `dataset/MRCLAM1/`.

## Prerequisites

The commands below use Linux with GCC, a C++17 compiler, CMake 3.21 or newer, and Python 3.10 or newer with development headers. The setup was checked on Ubuntu 22.04 with GCC 11 and Python 3.10. On Ubuntu or Debian, install the build tools with the following commands.

```sh
sudo apt-get update
sudo apt-get install build-essential cmake git ca-certificates pkg-config \
    libyaml-cpp-dev python3 python3-dev python3-venv
```

- **gsolver** uses OpenMP and yaml-cpp. Its CMake build fetches Eigen, the Rerun C++ SDK and its Apache Arrow dependency, and nanobind for Python. The first build requires network access. [gsolver on GitHub](https://github.com/rvp-group/gsolver/tree/18f2919f4d0468976bf80fbe5eeafda1637f33f0)
- **GTSAM** uses bundled Eigen, METIS, and pybind11. Its Python build requirements include NumPy, pyparsing, pytest, and pybind11-stubgen. The build below disables the optional Boost and TBB features. [GTSAM on GitHub](https://github.com/borglab/gtsam/tree/93859425e7501c925d1c6e86a0c819286ebf5653)

## Get the source

```sh
git clone --branch manifold --recurse-submodules https://github.com/JeongUk-K/GaBP_Tutorial.git
cd GaBP_Tutorial
```

The Git submodules pin both solver revisions. In an existing clone, run `git submodule update --init --recursive`. If you downloaded a GitHub ZIP, clone the solver sources into the extracted directory and check out the pinned commits.

```sh
git clone https://github.com/rvp-group/gsolver.git solver/gsolver
git -C solver/gsolver checkout --detach 18f2919f4d0468976bf80fbe5eeafda1637f33f0
git clone https://github.com/borglab/gtsam.git solver/gtsam
git -C solver/gtsam checkout --detach 93859425e7501c925d1c6e86a0c819286ebf5653
```

## Set up the Python virtual environment

Run the remaining commands from the project root.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy matplotlib Pillow imageio-ffmpeg
```

NumPy, Matplotlib, and Pillow support the Python examples and their GIF output. The dataset video script uses imageio-ffmpeg.
Follow the build section for each solver you plan to run.

### Build and install gsolver and gabp

```sh
cmake -S . -B build/gabp -DCMAKE_BUILD_TYPE=Release \
    -DGABP_BUILD_PYTHON=ON -DBUILD_TESTING=OFF \
    -DPython_EXECUTABLE="$VIRTUAL_ENV/bin/python"
cmake --build build/gabp --config Release --target gabp --parallel 2
cmake --install build/gabp --config Release --component gabp-python --prefix "$VIRTUAL_ENV"
python -c 'import gabp, gsolver; print(gabp.__file__); print(gsolver.__file__)'
```

The `gabp-python` install component places the two matching modules into `.venv`. If you use a separately checked-out copy of gsolver at the pinned revision, pass `-DGSOLVER_SOURCE_DIR=/absolute/path/to/gsolver` at configure time.

### Build and install GTSAM

```sh
python -m pip install -r solver/gtsam/python/dev_requirements.txt
cmake -S solver/gtsam -B build/gtsam -DCMAKE_BUILD_TYPE=Release \
    -DGTSAM_BUILD_PYTHON=ON -DGTSAM_BUILD_TESTS=OFF \
    -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF -DGTSAM_BUILD_UNSTABLE=OFF \
    -DGTSAM_UNSTABLE_BUILD_PYTHON=OFF -DGTSAM_WITH_TBB=OFF \
    -DGTSAM_USE_BOOST_FEATURES=OFF -DGTSAM_ENABLE_BOOST_SERIALIZATION=OFF \
    -DPython3_EXECUTABLE="$VIRTUAL_ENV/bin/python" \
    -DPYTHON_EXECUTABLE="$VIRTUAL_ENV/bin/python"
cmake --build build/gtsam --config Release --target python-install --parallel 2
python -c 'import gtsam; print(gtsam.__file__)'
```

This installs GTSAM's Python package into `.venv`. Keep `build/gtsam` in place because this source installation loads its native libraries from that directory. In a new shell, `source .venv/bin/activate` is enough to run either installed solver.

## Run the examples

```sh
OMP_NUM_THREADS=2 python "example/UTIAS data/gsolver/main.py"
python "example/UTIAS data/gtsam/main.py"
```

Edit the parameters at the top of each `main.py` to set `start_time`, `duration`, the sliding window length in seconds, the number of iterations per window, and the measurement covariances. `start_time` is an offset from the beginning of the common dataset interval, so `start_time + duration` must fit within 1487.796 seconds for Dataset 1. Each solver writes `iterations.gif`, `states.csv`, and `cost.csv` under its own `results/` directory.
Use the same dataset interval, window length, and planar factor covariances in both scripts for a solver comparison.

The gsolver examples also include two synthetic runs.

```sh
OMP_NUM_THREADS=2 python "example/toy_example/gsolver/multirobot_example.py"
OMP_NUM_THREADS=2 python "example/toy_example/gsolver/multirobot_sliding_winddow_example.py"
```

### Run a two-robot smoke example

```sh
OMP_NUM_THREADS=2 python - <<'PY'
import numpy as np
import gabp
import gsolver

def transform(x, y):
    matrix = np.eye(4)
    matrix[:2, 3] = (x, y)
    return matrix

def make_pose(node_id, t, x, y):
    state = gsolver.SE3PoseVelValue()
    state.pose = transform(x, y)
    state.vel = np.zeros(6)
    return gsolver.SE3PoseVel(node_id, state, 0.1 * np.eye(12), t)

graph = gabp.MultiRobotGraph(robot_count=2)
a0 = make_pose('a0', 0.0, 0.0, 0.0)
a1 = make_pose('a1', 1.0, 1.1, 0.0)
b1 = make_pose('b1', 1.0, 1.0, 2.2)
graph.add_pose(0, 0.0, a0)
graph.add_pose(0, 1.0, a1)
graph.add_pose(1, 1.0, b1)

prior = gsolver.SE3PoseVelPrior('prior', transform(0.0, 0.0), 0.01 * np.eye(6))
odom = gsolver.SE3PoseVelPoseVel('odom', transform(1.0, 0.0), 0.05 * np.eye(6))
observation = gabp.RangeBearingFactor('a1_sees_b1', np.array([2.0, np.pi / 2]), np.diag([0.05, 0.03]))
graph.add_factor(0, prior, a0)
graph.add_factor(0, odom, a0, a1)
graph.add_factor(0, observation, a1, b1)

graph.iterate()
graph.prune_before(1.0)
graph.iterate()
print('Robot 1 estimated XY:', b1.mu.pose[:2, 3])
PY
```

Robot indices start at zero. For an inter-robot observation, register the factor with the observing robot and pass the observer pose first. `prune_before()` removes poses strictly older than the cutoff and detaches their factors from every connected graph.

## Build the C++ libraries only

```sh
cmake -S . -B build-cpp -DCMAKE_BUILD_TYPE=Release \
    -DGABP_BUILD_PYTHON=OFF -DBUILD_TESTING=OFF
cmake --build build-cpp --config Release --target gabp_factors gabp_multirobot_graph --parallel 2
```

Link `gabp_factors` for the observation factors and `gabp_multirobot_graph` for the graph manager. Include `factors/range_factor.h`, `factors/range_bearing_factor.h`, and `multirobot/MultiRobotGraph.h` as needed.
