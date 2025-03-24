from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import subprocess

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path"""
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler."""
    import tempfile
    import os
    import setuptools

    with tempfile.NamedTemporaryFile('w', suffix='.cpp', delete=False) as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        fname = f.name
    try:
        compiler.compile([fname], extra_postargs=[flagname])
    except setuptools.distutils.errors.CompileError:
        return False
    finally:
        try:
            os.remove(fname)
        except OSError:
            pass
    return True

def cpp_flag(compiler):
    """Return the -std=c++[11/14/17] compiler flag.
    The newer version is preferred over c++11 (when it is available).
    """
    flags = ['-std=c++17', '-std=c++14', '-std=c++11']

    for flag in flags:
        if has_flag(compiler, flag): return flag

    raise RuntimeError('Unsupported compiler -- at least C++11 support is needed!')

# Compile the CUDA source file to object file
def compile_cuda(cuda_file, object_file):
    # Get conda environment paths
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    if not conda_prefix:
        raise RuntimeError("CONDA_PREFIX environment variable not set")
    
    # Get the C++ compiler from conda
    cxx = os.path.join(conda_prefix, 'bin', 'x86_64-conda-linux-gnu-g++')
    if not os.path.exists(cxx):
        raise RuntimeError(f"C++ compiler not found at {cxx}")
    
    # Get CUDA paths
    cuda_include = '/usr/local/cuda/include'
    cuda_lib = '/usr/local/cuda/lib64'
    
    # Compile CUDA file
    subprocess.check_call([
        'nvcc', '-c', 
        '-o', object_file, 
        f'-I{cuda_include}',
        f'-I{conda_prefix}/include',
        '-O3', 
        '-x', 'cu', 
        '-Xcompiler', '-fPIC',
        f'--compiler-bindir={cxx}',
        cuda_file
    ])
    return object_file

class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }

    if sys.platform == 'darwin':
        darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.7']
        c_opts['unix'] += darwin_opts

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')

        for ext in self.extensions:
            ext.define_macros = [('VERSION_INFO', '"%s"' % self.distribution.get_version())]
            ext.extra_compile_args = opts
            ext.extra_link_args = []
            if sys.platform == 'darwin':
                ext.extra_link_args = ['-stdlib=libc++']
            
            # Compile CUDA file to object file
            if hasattr(ext, 'cuda_sources') and ext.cuda_sources:
                cuda_obj_dir = os.path.join(self.build_temp, 'cuda_objs')
                os.makedirs(cuda_obj_dir, exist_ok=True)
                
                for cuda_source in ext.cuda_sources:
                    obj_name = os.path.splitext(os.path.basename(cuda_source))[0] + '.o'
                    object_file = os.path.join(cuda_obj_dir, obj_name)
                    compile_cuda(cuda_source, object_file)
                    ext.extra_objects.append(object_file)
                    
        build_ext.build_extensions(self)

# Define the extension
cuda_ext = Extension(
    'orange.cuda_ops',
    ['orange/cuda/bindings/bindings.cpp'],
    include_dirs=[
        get_pybind_include(),
        get_pybind_include(user=True),
        'orange/cuda/kernels',
        '/usr/local/cuda/include',
        os.path.join(os.environ.get('CONDA_PREFIX', ''), 'include')
    ],
    language='c++',
    extra_compile_args=['-O3'],
    extra_link_args=[],
    libraries=['cudart'],
    library_dirs=['/usr/local/cuda/lib64'],
    runtime_library_dirs=['/usr/local/cuda/lib64'],
    extra_objects=[],  # This will be filled with compiled CUDA objects
)

# Add CUDA sources
cuda_ext.cuda_sources = ['orange/cuda/kernels/tensor_ops.cu']

ext_modules = [cuda_ext]

setup(
    name='orange',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A deep learning framework with CUDA support',
    long_description='',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.4.3', 'numpy'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
) 