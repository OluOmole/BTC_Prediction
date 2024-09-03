from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as requirement_object:
        requirements=requirement_object.readlines()
        requirements=[requirement.replace("\n","") for requirement in requirements]

        if "-e ." in requirements:
            requirements.remove("-e .")
    
    return requirements

setup(
    name="bitcoin_price_prediction",
    version="0.1",
    description="A project for predicting Bitcoin price direction.",
    author="Oluwadamilare Omole",
    author_email="oluwadamilare.omole.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires= get_requirements("requirements.txt"),
    entry_points={
        "console_scripts": [
            "bitcoin_price_prediction=main:main",
        ],
    },
    
    #url="https://github.com/stevenomole/bitcoin_price_prediction",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)