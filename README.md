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

Add the cubitpy path to `PYTHONPATH`

```bash
export PYTHONPATH=path_to_cubitpy:$PYTHONPATH
```

Additionally paths to cubit and `pre_exodus` have to be set.

```bash
export BACI_PRE_EXODUS=path_to_pre-exodus
export CUBIT=path_to_cubit_directory
```

As an example; 
```bash
export CUBIT=/imcs/public/compsim/opt/cubit-13.2
export BACI_PRE_EXODUS=/home/a11btasa/git_repos/baci/baci_build/pre_exodus
export PYTHONPATH=/home/a11btasa/git_repos/cubitpy:$PYTHONPATH
```

# Running the BEM code

To run BEM implementation as the solver, make a syymbolic link in executables folder.

```bash
ln -s <your/path/to/BEM> <imcsml_dir>/config/baci-release
```

```bash
ln -s /home/a11btasa/git_repos/bem/bem /home/a11btasa/git_repos/imcsml/config/bem
```

# Running tests
TO check if the package/libraries are installed correctly tests are defined. Unittests for cubitpy and integration tests for pimcsml are created. To run the tests:

```bash
pytest
```
