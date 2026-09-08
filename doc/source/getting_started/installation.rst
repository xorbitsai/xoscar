.. _installation:

============
Installation
============

Xoscar can be installed via pip from `PyPI <https://pypi.org/project/xoscar>`__.

::

    pip install xoscar

Python version support
----------------------

Officially Python 3.10, 3.11, 3.12, 3.13 and 3.14.

Python 3.9 is no longer supported. Python 3.14 supports both the standard
GIL-enabled build and, experimentally, the free-threaded build (``3.14t``).
Free-threaded source builds require Cython 3.2 or newer and CMake 3.30 or newer.
Install with the free-threaded interpreter to select the matching ``cp314t``
wheel; regular ``cp314`` extension wheels are not ABI-compatible.

Actor and asyncio objects remain confined to their owning event loop. This
does not make sharing mutable actors, serialization inputs, or collective
contexts across threads safe without application-level synchronization.
Optional third-party extensions may re-enable the GIL; check
``sys._is_gil_enabled()`` after importing your application's dependencies.

.. versionadded:: v0.8.0
    Python 3.13 is supported since v0.8.0.

.. versionadded:: next release
    Support for standard GIL-enabled Python 3.14.

.. versionadded:: next release
    Experimental support for free-threaded Python 3.14 (``3.14t``).

.. versionchanged:: next release
    Python 3.9 is no longer supported; Python 3.10 is the minimum version.


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

Python 3.14 requires uvloop 0.22.1 or newer on non-Windows platforms.

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
