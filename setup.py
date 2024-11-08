import setuptools

VERSION = '0.2.8' 
DESCRIPTION = 'DeepMIMO'
LONG_DESCRIPTION = 'DeepMIMO dataset generator library'

# Setting up
setuptools.setup(
        name="DeepMIMO", 
        version=VERSION,
        author="Umut Demirhan, Ahmed Alkhateeb, João Morais",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        license_files = ('LICENSE.txt', ),
        install_requires=['numpy',
                          'scipy',
                          'tqdm',
                          'matplotlib',
                          ],
        
        keywords=['mmWave', 'MIMO', 'DeepMIMO', 'python', 'Beta'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent"
        ],
        package_dir={"": "src"},
        packages=setuptools.find_packages(where="src"),
        url='https://deepmimo.net/'
)
