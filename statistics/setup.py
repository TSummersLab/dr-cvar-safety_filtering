from setuptools import setup, find_packages

setup(
    name='statistics',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'cvxpy',
        'scipy',
        'matplotlib',
    ],
    author='Sleiman Safaoui',
    author_email='snsafaoui@gmail.com',
    description='statistics tools',
    url='',
)
