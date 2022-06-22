from setuptools import setup, find_packages
from pathlib import Path


def read_file(filename: str) -> str:
    with open(Path(__file__).parent / filename, mode='r', encoding='utf-8') as f:
        return f.read()


PACKAGE_NAME = 'fastxq'

__author__ = 'Vladimir Starostin'
__email__ = 'vladimir.starostin@uni-tuebingen.de'
__version__ = '0.0.1'
__license__ = 'MIT'

classifiers = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
]

description = 'Fast conversion of 2D X-ray diffraction images to reciprocal space'

long_description = read_file('README.md')
long_description_content_type = 'text/markdown'


entry_points = {
    'gui_scripts': [
        'gixi_client = gixi:main',
    ],
}


python_requires = '>=3.6'
install_requires = read_file('requirements.txt').splitlines()


setup(
    name=PACKAGE_NAME,
    packages=find_packages(),
    version=__version__,
    author=__author__,
    author_email=__email__,
    license=__license__,
    description=description,
    long_description=long_description,
    long_description_content_type=long_description_content_type,
    python_requires=python_requires,
    classifiers=classifiers,
    install_requires=install_requires
)
