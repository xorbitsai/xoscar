<div align="center">
  <img width="77%" alt="" src="https://raw.githubusercontent.com/xprobe-inc/xoscar/main/doc/source/_static/Xoscar.svg"><br>
</div>

# Python actor framework for heterogeneous computing.
[![PyPI Latest Release](https://img.shields.io/pypi/v/xoscar.svg?style=for-the-badge)](https://pypi.org/project/xoscar/)
[![Coverage](https://img.shields.io/codecov/c/github/xorbitsai/xoscar?style=for-the-badge)](https://codecov.io/gh/xorbitsai/xoscar)
[![Build Status](https://img.shields.io/github/actions/workflow/status/xorbitsai/xoscar/python.yaml?branch=main&style=for-the-badge&label=GITHUB%20ACTIONS&logo=github)](https://actions-badge.atrox.dev/xorbitsai/xoscar/goto?ref=main)
[![License](https://img.shields.io/pypi/l/xoscar.svg?style=for-the-badge)](https://github.com/xorbitsai/xoscar/blob/main/LICENSE)

## What is actor
Writing parallel and distributed programs is often challenging and requires a lot of time to deal with concurrency
issues. Actor model provides a high-level, scalable and robust abstraction for building distributed applications. 
It provides several benefits:
- Scalability: Actors easily scale across nodes. The asynchronous, non-blocking nature of actors allows them to handle huge volumes of concurrent tasks efficiently.
- Concurrency: The actor model abstracts over concurrency, allowing developers to avoid raw threads and locks.
- Modularity: An actor system decomposes naturally into a collection of actors that can be understood independently. Actor logic is encapsulated within the actor itself.

## Why Xoscar
Xoscar implements the actor model in Python and provides user-friendly APIs that offer significant benefits for building 
applications on heterogeneous hardware:
- **Abstraction over low-level communication details**: Xoscar handles all communication between actors transparently,
whether on CPUs, GPUs, or across nodes. Developers focus on application logic rather than managing hardware resources 
and optimizing data transfer.
- **Flexible actor models**: Xoscar supports both stateful and stateless actors. Stateful actors ensure thread safety for 
concurrent systems while stateless actors can handle massive volumes of concurrent messages. Developers choose the 
appropriate actor model for their needs.
- **Batch method**: Xoscar provides a batch interface to significantly improve call efficiency when an actor interface is 
invoked a large number of times.
- **Advanced debugging support**: Xoscar can detect potential issues like deadlocks, long-running calls, and performance
bottlenecks that would otherwise be nearly impossible to troubleshoot in a heterogeneous environment.
- **Automated recovery**: If an actor fails for any reason, Xoscar will automatically restart it if you want. It can monitor 
actors and restart them upon failure, enabling fault-tolerant systems.

## Overview
![architecture.png](doc/source/_static/architecture.png)
Xoscar allows you to create multiple actor pools on each worker node, typically binding an actor pool to a CPU core or 
a GPU card. Xoscar provides allocation policies so that whenever an actor is created, it will be instantiated in the
appropriate pool based on the specified policy.

When actors communicate, Xoscar will choose the optimal communication mechanism based on which pools the actors 
belong to. This allows Xoscar to optimize communication in heterogeneous environments with multiple processing 
units and accelerators.

## Where to get it

### PyPI
Binary installers for the latest released version are available at the [Python
Package Index (PyPI)](https://pypi.org/project/xoscar).

```shell
# PyPI
pip install xoscar
```

### Build from source
The source code is currently hosted on GitHub at: https://github.com/xorbitsai/xoscar .

Building from source requires that you have cmake and gcc installed on your system.

- cmake >= 3.11
- gcc >= 8

```shell
# If you have never cloned xoscar before
git clone --recursive https://github.com/xorbitsai/xoscar.git
cd xoscar/python
pip install -e .

# If you have already cloned xoscar before
cd xoscar
git submodule init
git submodule update
cd python && pip install -e .
```

## APIs
Here are basic APIs for Xoscar.
#### Define an actor
```python
import xoscar as xo

# stateful actor, for stateless actor, inherit from xo.StatelessActor
class MyActor(xo.Actor):
    def __init__(self, *args, **kwargs):
        pass

    async def __post_create__(self):
        # called after created
        pass

    async def __pre_destroy__(self):
        # called before destroy
        pass

    def method_a(self, arg_1, arg_2, **kw_1):  # user-defined function
        pass

    async def method_b(self, arg_1, arg_2, **kw_1):  # user-defined async function
        pass
```

#### Create an actor
```python
import xoscar as xo

actor_ref = await xo.create_actor(
    MyActor, 1, 2, a=1, b=2,
    address='<ip>:<port>', uid='UniqueActorName')
```

#### Get an actor reference
```python
import xoscar as xo

actor_ref = await xo.actor_ref(address, actor_id)
```

#### Invoke a method
```python
# send
await actor_ref.method_a.send(1, 2, a=1, b=2)
# equivalent to actor_ref.method_a.send
await actor_ref.method_a(1, 2, a=1, b=2)
# tell, it sends a message asynchronously and does not wait for a response.
await actor_ref.method_a.tell(1, 2, a=1, b=2)
```
### Batch method
Xoscar provides a set of APIs to write batch methods. You can simply add a `@extensible` decorator to your actor method
and create a batch version. All calls wrapped in a batch will be sent together, reducing possible RPC cost.
#### Define a batch method
```python
import xoscar as xo

class ExampleActor(xo.Actor):
    @xo.extensible
    async def batch_method(self, a, b=None):
        pass
```
Xoscar also supports creating a batch version of the method:
```python
class ExampleActor(xo.Actor):
    @xo.extensible
    async def batch_method(self, a, b=None):
        raise NotImplementedError  # this will redirect all requests to the batch version

    @batch_method.batch
    async def batch_method(self, args_list, kwargs_list):
        results = []
        for args, kwargs in zip(args_list, kwargs_list):
            a, b = self.batch_method.bind(*args, **kwargs)
            # process the request
            results.append(result)
        return results  # return a list of results
```
In a batch method, users can define how to more efficiently process a batch of requests.

#### Invoke a batch method
Calling batch methods is easy. You can use `<method_name>.delay` to make a batched call and use `<method_name>.batch` to send them:
```python
ref = await xo.actor_ref(uid='ExampleActor', address='127.0.0.1:13425')
results = await ref.batch_method.batch(
    ref.batch_method.delay(10, b=20),
    ref.batch_method.delay(20),
)
```

## License
[Apache 2](LICENSE)
