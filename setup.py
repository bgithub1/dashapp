"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='dashapp',  # Required
    version='0.0.1',  # Required
    description='Python wrapper for building Dash Web Apps',  # Required
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',  # Optional (see note above)
    url='https://github.com/bgithub1/dashapp.git',  # Optional
    author='Bill Perlman',  # Optional
    author_email='bperlman@liverisk.com',  # Optional
    classifiers=[  # Optional
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License'],
    python_requires='>=3.6',
    keywords='sample setuptools development',  # Optional
    packages=find_packages(),  # Required
#     install_requires=['psycopg2','pandasql','pandas','SQLAlchemy'],  # Optional
)
