# Setup

The required libraries are defined in environment.yml file and the following command creates the environment and installs them automatically.

```bash
conda env create
``` 

Next to activate the environment

```bash
conda activate imcsml
```

Finally setup the framework using python develop, which will install `pimcsml` and `cubitpy` packages.  

```bash
python setup.py develop
```

update the libraries 

```bash
conda env update
```

Additionally paths to cubit and `pre_exodus` have to be set (copy the following lines - indeed, with the correct paths - into your `.bashrc` file).

**Note:** Linking for the cubit requires the cubit directory and linking for baci_pre_exodus requires the executable itsel

```bash
export BACI_PRE_EXODUS=path_to_pre-exodus
export CUBIT=path_to_cubit_directory
```

As an example; 
```bash
export CUBIT=/imcs/public/compsim/opt/cubit-13.2
export BACI_PRE_EXODUS=/home/a11btasa/git_repos/baci/baci_build/pre_exodus
```

# Running the BACI code

To run Baci examples, a symbolic link should be done in executables folder.

- linking the `baci-release` executable--> which is the solver

```bash
ln -s <your/path/to/baci-release> <imcsml_dir>/executables/baci-release
```

- linking the `post_drt_ensight` executable --> which translates the baci output to to the paraview readable format

```bash
ln -s <your/path/to/post_drt_ensight> <imcsml_dir>/executables/post_drt_ensight
```

- linking the `post_processor` executable --> needed for `post_drt_ensight`

```bash
ln -s <your/path/to/post_processor> <imcsml_dir>/executables/post_processor
```



# Running the BEM code

To run BEM implementation as the solver, make a symbolic link in executables folder.

- linking the `bem` executable

```bash
ln -s <your/path/to/BEM-executable> <imcsml_dir>/executables/bem
```

# Running tests
To check if the program works without any problem, tests are generated. At this point `Python` offers the `pytest` interface. 

In `cubitpy`, `Unittesting` is offered while the `Integrationtesting` is generated for `pimcsml`. To run the tests:

```bash
pytest
```
If running all test scripts is not desired, then one can change the `testpaths` directory in the `setup.cfg` file for picking up specific tests.