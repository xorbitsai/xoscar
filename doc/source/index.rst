.. _index:

.. raw:: html

    <img class="align-center" alt="Xoscar Logo" src="_static/Xoscar.svg" style="background-color: transparent", width="77%">

====


Xoscar: Python actor framework for heterogeneous computing
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

What is actor model
-------------------
Writing parallel and distributed programs is often challenging and requires a lot of time to deal
with concurrency issues. Actor model provides a high-level, scalable and robust abstraction for
building distributed applications. It provides several benefits:

- Scalability: Actors easily scale across nodes. The asynchronous, non-blocking nature of actors
  allows them to handle huge volumes of concurrent tasks efficiently.

- Concurrency: The actor model abstracts over concurrency, allowing developers to avoid raw threads
  and locks.

- Modularity: An actor system decomposes naturally into a collection of actors that can be
  understood independently. Actor logic is encapsulated within the actor itself.


Why Xoscar
----------
Xoscar implements the actor model in Python and provides user-friendly APIs that offer significant
benefits for building applications on heterogeneous hardware:

- **Abstraction over low-level communication details**: Xoscar handles all communication between
  actors transparently, whether on CPUs, GPUs, or across nodes. Developers focus on application
  logic rather than managing hardware resources and optimizing data transfer.

- **Flexible actor models**: Xoscar supports both stateful and stateless actors. Stateful actors
  ensure thread safety for concurrent systems while stateless actors can handle massive volumes of
  concurrent messages. Developers choose the appropriate actor model for their needs.

- **Batch method**: Xoscar provides a batch interface to significantly improve call efficiency
  when an actor interface is invoked a large number of times.

- **Advanced debugging support**: Xoscar can detect potential issues like deadlocks, long-running
  calls, and performance bottlenecks that would otherwise be nearly impossible to troubleshoot in a
  heterogeneous environment.

- **Automated recovery**: If an actor fails for any reason, Xoscar will automatically restart it if
  you want. It can monitor actors and restart them upon failure, enabling fault-tolerant systems.

Overview
--------
.. image:: _static/architecture.png
   :alt: architecture

Xoscar allows you to create multiple actor pools on each worker node, typically binding an actor
pool to a CPU core or a GPU card. Xoscar provides allocation policies so that whenever an actor is
created, it will be instantiated in the appropriate pool based on the specified policy.

When actors communicate, Xoscar will choose the optimal communication mechanism based on which
pools the actors belong to. This allows Xoscar to optimize communication in heterogeneous
environments with multiple processing units and accelerators.

Where to get it
---------------
The source code is currently hosted on GitHub at: https://github.com/xorbitsai/xoscar

Binary installers for the latest released version are available at the
`Python Package Index (PyPI) <https://pypi.org/project/xoscar/>`_.

::

   # PyPI
   pip install xoscar

License
-------
`Apache 2 <https://github.com/xorbitsai/xoscar/blob/main/LICENSE>`_

.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started/index
   user_guide/index
   reference/index
