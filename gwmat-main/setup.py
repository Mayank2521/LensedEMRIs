from setuptools import setup, find_packages
from setuptools import Extension
from Cython.Build import cythonize

# Define Cython extensions
extensions = [
    Extension("gwmat.cythonized_point_lens", ["gwmat/cythonized_point_lens.pyx"])
]

setup(
    name='gwmat',
    version='1.0.0',
    description='A package for generating and analysing microlensed gravitational wave signals.',
    author='Anuj Mishra',
    author_email='anuj.mishra@ligo.org',
    url='https://git.ligo.org/anuj.mishra/gwmat/',
    packages=find_packages(),
    package_data={
        "gwmat": ["detector_PSDs/*.txt"],
    },
    ext_modules=cythonize(extensions),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    python_requires='>=3.6',
    license='MIT',
    package_dir={'gwmat': 'gwmat'},
)
