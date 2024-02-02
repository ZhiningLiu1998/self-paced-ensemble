#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev

# %%
import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# %%

# get __version__ from _version.py
ver_file = os.path.join("self_paced_ensemble", "__version__.py")
with open(ver_file) as f:
    exec(f.read())

# Package meta-data.
NAME = 'self-paced-ensemble'
DESCRIPTION = 'Self-paced Ensemble for classification on class-imbalanced data.'
URL = 'https://github.com/ZhiningLiu1998/self-paced-ensemble'
PROJECT_URLS = {
    'Documentation': 'https://imbalanced-ensemble.readthedocs.io/en/latest/api/ensemble/_autosummary/imbalanced_ensemble.ensemble.under_sampling.SelfPacedEnsembleClassifier.html',
    'Source': 'https://github.com/ZhiningLiu1998/self-paced-ensemble',
    'Tracker': 'https://github.com/ZhiningLiu1998/self-paced-ensemble/issues',
    'Download': 'https://pypi.org/project/self-paced-ensemble/#files',
}
EMAIL = 'zhining.liu@outlook.com'
AUTHOR = 'Zhining Liu'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = __version__
LICENSE = "MIT"
REQUIRED = [
    "numpy>=1.13.3",
    "pandas>=1.1.3",
    "scipy>=0.19.1",
    "scikit-learn>=0.24",
    "joblib>=0.11",
    "imbalanced-learn>=0.7.0",
    "imbalanced-ensemble>=0.2.1",
]
EXTRAS = {
}
CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved",
    "Programming Language :: C",
    "Programming Language :: Python",
    "Topic :: Software Development",
    "Topic :: Scientific/Engineering",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
]

# %%

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()


# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    project_urls=PROJECT_URLS,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],

    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license=LICENSE,
    classifiers=CLASSIFIERS,
    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand,
    },
)

# %%
