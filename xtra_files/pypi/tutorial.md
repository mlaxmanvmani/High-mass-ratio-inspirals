### [Packaging Python Projects](https://packaging.python.org/en/latest/tutorials/packaging-projects/#packaging-python-projects)

* Website (same link as the headline):
    https://packaging.python.org/en/latest/tutorials/packaging-projects/#packaging-python-projects
* This assumes that the project already contains `README.md` and `LICENSE` files.

First, upgrade `pip`
```
python3 -m pip install --upgrade pip
```

Next, a `pyproject.toml` is needed,  e.g. for `gwbench` see `xtra_files/pypi/pyproject.toml`.
- https://gitlab.com/sborhanian/gwbench/-/blob/master/xtra_files/pypi/pyproject.toml

Next, we install/upgrade `build` and `twine`
```
python3 -m pip install --upgrade build twine
```
and then `build` the package to generate archive files in the `dist` directory
```
python3 -m build
```

Next, generate an [API token](https://pypi.org/manage/account/#api-tokens) on PYPI. The use `twine` to upload all of the archives in `dist` using `__token__` as the username and the the API token for the password.

```
python3 -m twine upload --repository pypi dist/*
```
