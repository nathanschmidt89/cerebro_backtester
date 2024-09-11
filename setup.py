from setuptools import setup, find_packages

setup(
    name='cerebro_backtester',
    version='10.0.1',
    author='Nathan Schmidt',
    description='',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/nathanschmidt89/cerebro_backtester',
    packages=find_packages(),
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Unlicense",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.12',
)