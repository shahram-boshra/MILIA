import os
from setuptools import setup, find_packages

setup(
    name="milia-pipeline",
    version="1.0.0",
    packages=find_packages(),
    
    # Empty - all dependencies managed by conda via environment.yml
    install_requires=[],
    
    extras_require={
        'dev': []
    },
    
    python_requires=">=3.10",
    
    author="Asadollah (Shahram) Boshra, Ilia Boshra",
    author_email="a.boshra@gmail.com, ilia.boshra@gmail.com",  
    
    description="Milia molecular graph dataset processing pipeline for quantum chemistry ML",
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type="text/markdown",
    
    url="https://github.com/yourusername/milia-pipeline",
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
    
    entry_points={
        'console_scripts': [
            'milia=main:main',
        ],
    },
    
    package_data={
        'milia_pipeline': ['*.yaml', '*.yml'],
    },
    include_package_data=True,
)
