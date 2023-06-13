.. _llm_inference:

=================================================
Use case: LLM inference on heterogeneous hardware
=================================================

As the power of open-source LLMs continues to grow, it becomes important to consider their
integration into production workflows. Leveraging the actor model and the potential of
heterogeneous hardware, Xoscar presents an ideal choice for private LLM inference.

While GPUs are well-suited for training and inference, their resources can be expensive and
limited. However, GPU servers often come with powerful CPUs and ample main memory. This raises
the question: why not offload less critical inference tasks to CPUs?

This tutorial will guide you through the process of harnessing both GPUs and CPUs for LLM inference
using Xoscar.


Setup
-----
Before proceeding, ensure that you have installed the CUDA Toolkit by following the
`installation guide <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#open-ubuntu-installation>`_.

To verify your installation, run::

   nvcc --version

You should receive an output similar to:

.. code-block:: text

   nvcc: NVIDIA (R) Cuda compiler driver
   Copyright (c) 2005-2022 NVIDIA Corporation
   Built on Tue_Mar__8_18:18:20_PST_2022
   Cuda compilation tools, release 11.6, V11.6.124
   Build cuda_11.6.r11.6/compiler.31057947_0

Once confirmed, proceed to install other necessary dependencies::

   pip install xoscar
   CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python

For this introduction, we will be using the LLM
`vicuna-13B <https://huggingface.co/vicuna/ggml-vicuna-13b-1.1>`_. However, feel free to choose any
LLM that suits your requirements, making sure to adapt the system prompt accordingly.


Define actors
-------------

Let's import :code:`xoscar` at the very beginning:

.. code-block:: python

   import asyncio
   import xoscar as xo

   from typing import List, Optional, Dict, Any


The actor leverages the underlying implementation in llama.cpp, with the keyword argument
``n_gpu_layers`` determining the number of layers loaded into VRAM. By setting ``n_gpu_layers`` to
``0``, the model will be loaded into main memory, enabling pure CPU-based inference.

.. code-block:: python

   class InferenceActor(xo.Actor):

       # choose the system prompt that best fits your model.
       system_prompt: str = (
           "A chat between a curious user and an artificial intelligence assistant. The assistant"
           " gives helpful, detailed, and polite answers to the user's questions. ### User: %s ###"
       )

       def __init__(self, model_path: str, *args, **kwargs):
           from llama_cpp import Llama

           self._llm = Llama(model_path=model_path, *args, **kwargs)

       def inference(self, prompt: str, *args, **kwargs) -> Dict[str, Any]:
           return self._llm(self.system_prompt % prompt, *args, **kwargs)

.. seealso::
   :ref:`actor`


Create actor pools
------------------

Consider creating an actor pool for CPU inference, along with an actor pool for each GPU device.
In this tutorial, we have one GPU device, resulting in the creation of two actor pools.

Once the actor pools are successfully created, gather the address of each pool for future steps.

.. code-block:: python

   async def create_actor_pools(
       address: str,
       n_process: int,
       visible_cuda_device_idx: Optional[List[int]] = None,
   ) -> "xo.MainActorPoolType":
       assert n_process > 0

       envs = []
       if visible_cuda_device_idx is not None:
           assert len(visible_cuda_device_idx) == n_process
           for i in range(n_process):
               envs.append({"CUDA_VISIBLE_DEVICES": str(visible_cuda_device_idx[i])})

       return await xo.create_actor_pool(
           address=address,
           n_process=n_process,
           envs=envs
       )

   # set the environment variable CUDA_VISIBLE_DEVICES to -1 to prevent CPU actor pool
   # from using CUDA devices.
   await create_actor_pools("localhost:9999", n_process=2, visible_cuda_device_idx=[0, -1])

   pool_config = await xo.get_pool_config("localhost:9999")
   pool_addresses = pool_config.get_external_addresses()

.. seealso::
   :ref:`actor-pool`

Create actors
-------------

Now, it's time to create the actors. We will create two actors: one running purely on CPUs and the
other accelerated by GPU.

.. code-block:: python

   gpu_inference_actor = await xo.create_actor(
       InferenceActor,
       address=pool_addresses[1],
       uid="gpu",
       model_path="/path/to/ggml-vic7b-uncensored-q5_1.bin",
       n_gpu_layers=32
   )
   cpu_inference_actor = await xo.create_actor(
       InferenceActor,
       address=pool_addresses[2],
       uid="cpu",
       model_path="/path/to/ggml-vic7b-uncensored-q5_1.bin"
   )


Inference
---------

Invoke the actors to perform inference.

.. code-block:: python

   tasks = []
   tasks.append(
       gpu_inference_actor.inference(
           prompt="Define heterogeneous computing.",
           max_tokens=256,
           stop=["###"],
           echo=False
       )
   )
   tasks.append(
       cpu_inference_actor.inference(
           prompt="Define actor model.",
           max_tokens=256,
           stop=["###"],
           echo=False
       )
   )

   results = await asyncio.gather(*tasks)

And here are the results (after formatting):

::

   ### User: Define heterogeneous computing.
   ### Assistant: Heterogeneous computing refers to a computing system that consists of multiple processing units with different architectures and instruction sets, working together to perform computational tasks. In such systems, different types of processors are used to handle different parts of a computation, taking advantage of their unique strengths to improve overall performance. This can include processors with different instruction sets, such as GPUs (graphics processing units) and TPUs (tensor processing units), which are designed for specific types of computations, as well as CPUs (central processing units) that handle more general-purpose tasks. Heterogeneous computing is used in a wide range of applications, from high-performance computing to machine learning and artificial intelligence. It allows for greater flexibility and efficiency in the use of computational resources, and can lead to significant performance improvements over homogeneous (single-type) computing systems.

::

   ### User: Define actor model.
   ### Assistant: The Actor Model is a mathematical model for concurrency that describes computation as the sending of messages between actors, which represent independent units of concurrent computation. It was introduced by Carl Hewitt in 1973 and has since become an influential concept in the field of computer science, particularly in the development of concurrent and distributed systems. In the Actor Model, each actor has a unique identity, known as its "name," and can send and receive messages to and from other actors. Actors are isolated from one another and have no knowledge of the actions of other actors, which simplifies the design of concurrent programs by reducing the need for synchronization mechanisms. The Actor Model has been applied in a variety of contexts, including software engineering, distributed systems, and concurrency theory.

Conclusion
----------
In conclusion, Xoscar empowers you to seamlessly integrate private LLMs into your production
workflows, effectively leveraging the full potential of your heterogeneous hardware.