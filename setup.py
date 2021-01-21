import setuptools

# with open('README.md', 'r') as f:
#     long_description = f.read()

setuptools.setup(
    name='dmpy',
    version='0.1.0',
    author='Kim Lindberg Schwaner',
    author_email='kils@mmmi.sdu.dk',
    description='DMPy - Dynamic Motor Primitives in Python',
    # long_description=long_description,
    # long_description_content_type='text/markdown',
    # url='https://github.com/pypa/sampleproject',
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)
