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

## Contents
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
cd <deep_sim-repo-directory>
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

# Working with Cubitpy

To generate FEM models for Baci simulations, Cubitpy interface is used and Cubitpy requires `pre_exodus` executable and the `cubit` folder. 
To integrate cubitpy, the following paths are exported.

```bash
export BACI_PRE_EXODUS=path_to_pre-exodus
export CUBIT=path_to_cubit_directory
```

# Running the BACI code

To generate FEM models for Baci simulations, `pre_exodus` executable has to be accessible. Thus, `pre_exodus` should be exported as an environment variable.

```bash
export BACI_PRE_EXODUS=<path_to_pre_exodus>
```

To run Baci simulations, `baci-release` and `post_drt_ensight` executables should be exported as environment variables.

```bash
export BACI_RELEASE=<path_to_baci-release>
```

```bash
export BACI_POST_DRT_ENSIGHT=<path_to_post_drt_ensight>
```

# Running the BEM code

To run BEM implementation as the solver, in the same manner `bem` exectuable should be exported as an environment variable.

```bash
export BACI_POST_DRT_ENSIGHT=<path_to_bem>
```

# Running tests
To check if the program works without any problem, tests are generated. At this point `Python` offers the `pytest` interface. 

In `cubitpy`, `Unittesting` is offered while the `Integrationtesting` is generated for `pydeep_sim`. To run the tests:

```bash
pytest
```
If running all test scripts is not desired, then one can change the `testpaths` directory in the `setup.cfg` file for picking up specific tests.