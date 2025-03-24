from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
import sys
import os
import subprocess

class CMakeExtension(build_ext):
    def __init__(self, name, sourcedir=''):
        build_ext.__init__(self)
        self.name = name
        self.sourcedir = os.path.abspath(sourcedir)

    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                     '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version())

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                            cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                            cwd=self.build_temp)

setup(
    name='orange',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A PyTorch-like Autograd framework with CUDA support',
    long_description='',
    packages=find_packages(),
    ext_modules=[CMakeExtension('orange.cuda_ops')],
    cmdclass=dict(build_ext=CMakeExtension),
    zip_safe=False,
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.19.0',
        'pybind11>=2.6.0',
    ],
) 