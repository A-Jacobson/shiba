from setuptools import setup
# to deploy
# python setup.py sdist bdist_wheel upload
setup(
    name='shiba',
    version='0.0.2',
    description='PyTorch trainer, datasets, criterion, and utility functions',
    url='https://github.com/A-Jacobson/shiba',
    license='MIT',
    python_requires='>=3',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.6'
    ],
    packages=['shiba']
)
