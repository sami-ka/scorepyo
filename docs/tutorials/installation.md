# Installation

## From pip

The best way to use this package is to install it via pip in a conda environment, with python 3.8 or 3.9. 

First create your environment, with the environment name of your choice:
```shell
conda create -n <env_name> python=3.9
```

Then activate the environment: 

```shell
conda activate <env_name>
```

and finally install scorepyo via pip

```shell
pip install scorepyo
```

## From source

If you want to use the latest development from scorepyo or contribute to the package, you should install it from Github.



You still have to create a conda environment beforehand :
```shell
conda create -n <env_name> python=3.9
```

Then activate the environment: 

```shell
conda activate <env_name>
```

Clone the repository:

```shell
git clone https://github.com/drskd/scorepyo.git
```

Go to newly created scorepyo folder and launch the `create_env` function from `codepal.py`:
```shell
cd scorepyo

python codepal.py create_env -e <env_name>

```