# DeepSim

Deep_Sim is a package to automatize the simulations (FEM and BEM) and to predict the quantities using Machine Learning and Deep Learning techniques.  

--------------------------------------------------------------------------------------

                            *****    ******* *******  *****       *******  ** ***     ***
                            **  ***  **      **       **  **      **       ** ****   ****
                            **   *** *****   *****    *****       *******  ** ** ** ** **
                            **  ***  **      **       **               **  ** **  ***  **
                            *****    ******* *******  **          *******  ** **   *   **
--------------------------------------------------------------------------------------
                                        A Multipurpose Python automatization tool
                                        to run BEM and FEM simulations and apply
                                          Machine and Deep Learning techniques 
-------------------------------------------------------------------------------------

# Contents
1. [Prerequisites](#prerequisites)
1. [Installation](#installation)

## Prerequisites


Deep_Sim is developed with `python3.8`.
It is recommended to use virtual environments with `python`. Thus, the
`python3-venv` package can be installed using the following command (On Debian systems):
```bash
sudo apt-get install python3-venv python3-dev
```

## Installation

Create a new virtual environment using `venv` (for example in the home directory)
```bash
cd ~
mkdir opt
cd opt
python3 -m venv deep_sim_env
```

Activate the virtual environment:
```bash
source ~/opt/deep_sim_env/bin/activate
```

Go to the repository directory where you cloned to install the packages:
```bash
cd <deep_sim-directory>
```

Run the following command to install the default required packages. These packages are defined in the `requirements.txt` folder.
```bash
pip install -r requirements.txt
```

Finally setup the framework using python develop, which will install `pydeep_sim` and `cubitpy` packages.  

```bash
python setup.py develop
```

List the installed packages

```bash
pip list --local
```

# Working with CubitPy

To generate FEM models for Baci simulations, `CubitPy` interface is used. Since CubitPy is another Git repo and it should be added as a submodule for `deep_sim`:

```bash
git submodule add git@gitlab.com:compsim/codes/cubitpy.git
```

Add the cubitpy path to PYTHONPATH

```bash
export PYTHONPATH=<path_to_cubitpy>:$PYTHONPATH
```

Cubitpy requires `pre_exodus` executable and the `cubit` folder so set the paths for them:

```bash
export BACI_PRE_EXODUS=<path_to_pre-exodus>
export CUBIT=<path_to_cubit_directory>
```

Install the packages required for cubit:

```bash
pip install -r path_to_cubitpy/requirements.txt
```

# Running the BACI code

To run Baci simulations, `baci-release` and `post_drt_ensight` executables should be exported as environment variables.

```bash
export BACI_RELEASE=<path_to_baci-release>
```

```bash
export BACI_POST_DRT_ENSIGHT=<path_to_post_drt_ensight>
```
## Exporting results with Paraview
Since the simulation results of `BACI` have a special format, it is not possible to get the results without 
using Paraview or a post processing tool. Thus, it is possible to integrate Paraview to DeepSim. To do that

 - export the path to PARAVIEW, 
```bash
export PARAVIEWPATH=<path_to_paraview>
```
 - add the site packages to PYTHONPATH,
```bash
export PYTHONPATH=$PYTHONPATH:$PARAVIEWPATH:/lib/python2.7/site-packages
```
 - add the libraries of Paraview to LD_LIBRARY_PATH which looks for dynamic/shared libraries,
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PARAVIEWPATH}/lib
```

# Running the BEM code

To run BEM implementation as the solver, in the same manner `bem` exectuable should be exported as an environment variable.

```bash
export BACI_POST_DRT_ENSIGHT=<path_to_bem>
```

# Running tests
To check if the program works without any problem, tests are generated. At this point `Python` offers the `pytest` interface. To run the tests:

```bash
pytest
```
If running all test scripts is not desired, then one can change the `testpaths` directory in the `setup.cfg` file for picking up specific tests.

# Optional - But recommended 

If you are a VSCode user and you don't want to export `PYTHONPATH` and all env variables all the time since you didn't export them permanently, i.e. `.bashrc` and more important you don't want to. There is a super easy way.

- Add the following lines to `.vscode/settings.json`: (Indeed, change the paths depending on your setup)

```json
"terminal.integrated.env.linux": {
        "PARAVIEWPATH" : "/imcs/public/compsim/opt/ParaView-5.9.1-MPI-Linux-Python3.8-64bit",
        "PYTHONPATH": "${env:PYTHONPATH}:/home/a11btasa/deep_sim/cubitpy/:/imcs/public/compsim/opt/ParaView-5.9.1-MPI-Linux-Python3.8-64bit/lib/python3.8/site-packages",
        "LD_LIBRARY_PATH" : "${env:LD_LIBRARY_PATH}:/imcs/public/compsim/opt/ParaView-5.9.1-MPI-Linux-Python3.8-64bit/lib",
        "BACI_RELEASE" : "/home/a11btasa/git_repos/baci/baci_build/baci-release",
        "BACI_POST_DRT_ENSIGHT" : "/home/a11btasa/git_repos/baci/baci_build/post_drt_ensight",
        "CUBIT": "/imcs/public/compsim/opt/cubit-13.2",
        "BACI_PRE_EXODUS": "/home/a11btasa/git_repos/baci/baci_build/pre_exodus",
        "BEM" : "/home/a11btasa/git_repos/bem/bem"
    },
"python.envFile": "${workspaceFolder}/debug.env",
```

- Create the `debug.env` file where you clone the repo and copy the following lines there: (Indeed, again change the paths)
```bash
PARAVIEWPATH=/imcs/public/compsim/opt/ParaView-5.9.1-MPI-Linux-Python3.8-64bit
PYTHONPATH=${PYTHONPATH}:/home/a11btasa/deep_sim/cubitpy/:/imcs/public/compsim/opt/ParaView-5.9.1-MPI-Linux-Python3.8-64bit/lib/python3.8/site-packages
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/imcs/public/compsim/opt/ParaView-5.9.1-MPI-Linux-Python3.8-64bit/lib
BACI_RELEASE=/home/a11btasa/git_repos/baci/baci_build/baci-release
BACI_POST_DRT_ENSIGHT=/home/a11btasa/git_repos/baci/baci_build/post_drt_ensight
CUBIT=/imcs/public/compsim/opt/cubit-13.2
BACI_PRE_EXODUS=/home/a11btasa/git_repos/baci/baci_build/pre_exodus
BEM=/home/a11btasa/git_repos/bem/bem
```

The `settings.json` will set the variables for the integrated shell, while `debug.env` will set the variable but for the debugging menu and they must be done seperately. 

**Note:** If you are not a linux user, eg. windows, then set `terminal.integrated.env.windows`.

---
## References:
- Environment [variables](https://code.visualstudio.com/docs/python/environments#_environment-variable-definitions-file) in VSCODE