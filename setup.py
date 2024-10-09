from setuptools import setup, find_packages

setup(
    name='gwbench',
    version='0.8.4',
    author='Ssohrab Borhanian',
    author_email='sborhanian@gmail.com',
    description='A small tool to benchmark GW events',
    url='https://gitlab.com/sborhanian/gwbench',
    packages = find_packages(),
    package_data = {
        'gwbench.noise_curves': ['*.txt'],
                   },
    classifiers=[
        'Programming Language :: Python :: 3',
                ],
    python_requires='>=3.9',
      )
