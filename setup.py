from setuptools import setup, Extension
from ShallowTree import __version__ as version

with open('README.md', 'r') as f:
    long_description = f.read()

#  https://stackoverflow.com/questions/4529555/building-a-ctypes-based-c-library-with-distutils
from distutils.command.build_ext import build_ext as build_ext_orig
class CTypesExtension(Extension): pass
class build_ext(build_ext_orig):

    def build_extension(self, ext):
        self._ctypes = isinstance(ext, CTypesExtension)
        return super().build_extension(ext)

    def get_export_symbols(self, ext):
        if self._ctypes:
            return ext.export_symbols
        return super().get_export_symbols(ext)

    def get_ext_filename(self, ext_name):
        if self._ctypes:
            return ext_name + '.so'
        return super().get_ext_filename(ext_name)


setup(
    name="ShallowTree",
    version=version,
    author="Lucas Murtinho",
    author_email="lucas.murtinho@gmail.com",
    license="MIT",
    description="Shallow decision trees for explainable clustering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lmurtinho/ShallowTree",
    py_modules=['ShallowTree/ShallowTree',
                'ShallowTree/BisectionTree',
                'ShallowTree/RandomThresholdTree'],
    ext_modules=[CTypesExtension('ShallowTree/lib_best_cut', 
                                ['ShallowTree/best_cut.c'])],
    install_requires=['numpy', 'scikit-learn', 'ExKMC'],
    cmdclass={'build_ext': build_ext},
)