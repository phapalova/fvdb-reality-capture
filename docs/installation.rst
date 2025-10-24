Installing fVDB-Reality-Capture
================================================================

fVDB-Reality-Capture depends on `fVDB <https://fvdb.ai>`_ which in turn depends on `PyTorch <https://pytorch.org/>`_,
and requires a CUDA-capable GPU. Below are the supported sofware and hardware configurations.

Software Requirements
------------------------

fVDB is currently supported on the matrix of dependencies in the following table.

+------------------+-----------------+-----------------+----------------+------------------------------------------+
| Operating System | PyTorch Version | Python Version  | CUDA Version   | Vulkan Version (only for visualization)  |
+------------------+-----------------+-----------------+----------------+------------------------------------------+
| Linux Only       | 2.9.0           | 3.10 - 3.13     | 12.8 - 13.0    | 1.3.275.0                                |
+------------------+-----------------+-----------------+----------------+------------------------------------------+

Driver and Hardware Requirements
-----------------------------------

The following table specifies the minimum NVIDIA driver versions and GPU architectures needed to run fVDB-Reality-Capture:

+------------------+----------------+------------------+---------------------+
| Operating System | Driver Version | GPU Architecture | Comptue Capability  |
+------------------+----------------+------------------+---------------------+
| Linux Only       | 550.0 or later | Ampere or later  | 8.0 or greater      |
+------------------+----------------+------------------+---------------------+


Installation from pre-built wheels
-------------------------------------
To get started, run the appropriate pip install command for your Pytorch/CUDA versions. This command will install
the correct version of `fvdb-core` if it is not already installed.


PyTorch 2.9.0 + CUDA 12.8
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pip install fvdb-reality-capture fvdb-core==0.3.0+pt29.cu128 --extra-index-url="https://d36m13axqqhiit.cloudfront.net/simple" torch==2.9.0 --extra-index-url https://download.pytorch.org/whl/cu128


PyTorch 2.9.0 + CUDA 13.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pip install fvdb-reality-capture fvdb-core==0.3.0+pt29.cu130 --extra-index-url="https://d36m13axqqhiit.cloudfront.net/simple" torch==2.9.0 --extra-index-url https://download.pytorch.org/whl/cu130




Installation from source
-----------------------------

Clone the [fvdb-core repository](https://github.com/openvdb/fvdb-core) and the [fvdb-reality-capture repository](https://github.com/openvdb/fvdb-reality-capture).

.. code-block:: bash

   git clone git@github.com:openvdb/fvdb-core.git
   git clone git@github.com:openvdb/fvdb-reality-capture.git

Next build and install the fVDB library

.. code-block:: bash

   pushd fvdb-core
   ./build.sh install verbose editor_force
   popd

Finally, install fVDB-Reality-Capture

.. code-block:: bash

    pushd fvdb-reality-capture
    pip install .
    popd
