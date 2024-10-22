from setuptools import setup,find_packages
from typing import List
import os

HYPHEN_E_DOT = "-e ."

def get_requirements(file_path:str) -> List[str]:
    requirements = []
    with open(file_path,'r') as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n','') for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
    author = "ROSHAN KUMAR SAHU",
    author_email= "igniterofficial909505@gmail.com",
    version="1.0.0",
    name="age_and_gender_prediction",
    description="A Python package for predicting age and gender from an image",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Igniter-909/age_and_gender_prediction",
    packages = find_packages(),
    install_requires = get_requirements("requirements.txt")
)

