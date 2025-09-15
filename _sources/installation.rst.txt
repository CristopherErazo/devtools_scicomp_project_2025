Installation
============

From GitHub
-----------
To install the Spin Sampler library, follow these steps:

1. Clone the repository:

   .. code-block:: bash

       git clone https://github.com/CristopherErazo/devtools_scicomp_project_2025.git <folder_name>

2. Navigate to the folder directory, create and activate a virtual environment (with `python 3.9` preferably):

   .. code-block:: bash

        conda create --name <name> python=3.9
        conda activate <name>


3. Install the required dependencies:

   .. code-block:: bash

       pip install -r requirements.txt

4. Precompile the ``numba`` module:

   .. code-block:: bash

       python src/spin_sampler/compile_gibbs.py

This creates a file (`.so` in linux or `.pyd` in Windows) that contains the precompiled version of the `numba` functions and can be called as a module. If running on Windows you might need to install `MSVC Build Tools <https://visualstudio.microsoft.com/visual-cpp-build-tools/>`_ first, selecting *Desktop development with C++* during installation.
    

5. Install the package:

   .. code-block:: bash

       pip install .

Tests
-----

You can run the unit tests just by running:

    .. code-block:: bash

        pytest