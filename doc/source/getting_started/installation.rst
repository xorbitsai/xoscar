.. _installation:

============
Installation
============

Xoscar can be installed via pip from `PyPI <https://pypi.org/project/xoscar>`__.

::

    pip install xoscar

Python version support
----------------------

Officially Python 3.9, 3.10, 3.11, 3.12 and 3.13.

.. versionadded:: v0.8.0
    Python 3.13 is supported since v0.8.0.


Dependencies
------------

================================================================ ==========================
Package                                                          Minimum supported version
================================================================ ==========================
`NumPy <https://numpy.org>`__                                    1.20.3
`pandas <https://pandas.pydata.org>`__                           1.0.0
`scipy <https://scipy.org>`__                                    1.0.0
`scikit-learn <https://scikit-learn.org/stable>`__               0.20
cloudpickle                                                      1.5.0
psutil                                                           5.9.0
uvloop (for systems other than win32)                            0.21.0
================================================================ ==========================

Event loop selection
--------------------

Actor pool subprocesses use uvloop when it is installed by default. Set
``XOSCAR_USE_UVLOOP=0`` (or ``false``) to use asyncio's default event loop,
or ``XOSCAR_USE_UVLOOP=1`` (or ``true``) to require uvloop. The default value,
``auto``, detects whether uvloop is available. An empty or whitespace-only value
is treated as ``auto``. Values are case-insensitive; invalid values raise
``ValueError``.

An explicit ``create_actor_pool(..., use_uvloop=True)`` or ``use_uvloop=False``
takes precedence over the environment variable. The default
``use_uvloop="auto"`` reads the environment variable when creating the pool.
New subprocesses added with ``append_sub_pool`` inherit the pool's resolved
setting unless explicitly overridden. These settings do not change the main
process's existing event loop, which is controlled by the calling application.

.. warning::

   Upgrading to the uvloop selection fix changes effective default behavior on
   Linux and macOS. Previously, subprocess startup replaced the selected uvloop
   with asyncio's default loop; now ``auto`` actually uses the installed uvloop.
   To preserve the previous runtime behavior while validating your application,
   set ``XOSCAR_USE_UVLOOP=0`` or pass ``use_uvloop=False`` explicitly.
