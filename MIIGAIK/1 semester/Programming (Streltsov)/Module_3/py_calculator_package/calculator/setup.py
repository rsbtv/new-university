from setuptools import setup, find_packages

setup(
    name='calculator',
    version='1.0',
    description='Calculator',
    author='Rafael Sabitov',
    author_email='rafael.sabitov0703@gmail.com',
    packages=find_packages(),
    install_requires=[
        'math'
    ]
)