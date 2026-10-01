"""Compare socket/UCX/NIXL copy_to across two GPU worker processes.

Example: python benchmark_copy_to.py --backend nixl --iterations 30
Run each backend in a fresh process because the wheels may bundle different UCX versions.
"""

import argparse
import asyncio
import json
import statistics
import time

import xoscar as xo
from xoscar.backends.allocate_strategy import ProcessIndex


class TransferActor(xo.Actor):
    def __init__(self, device, array_type):
        self.device = device
        self.array_type = array_type
        self.buffers = []

    def allocate(self, sizes, value=0):
        if self.array_type == "torch":
            import torch

            torch.cuda.set_device(self.device)
            self.buffers = [
                torch.full(
                    (size,), value, dtype=torch.uint8, device=f"cuda:{self.device}"
                )
                for size in sizes
            ]
        else:
            import cupy as cp

            cp.cuda.Device(self.device).use()
            self.buffers = [cp.full(size, value, dtype=cp.uint8) for size in sizes]
        return [xo.buffer_ref(self.address, b) for b in self.buffers]

    def verify(self, value):
        return all(
            bool(
                (
                    (b.cpu().numpy() if self.array_type == "torch" else b.get())
                    == value
                ).all()
            )
            for b in self.buffers
        )

    async def measure(self, target, sizes, iterations):
        self.allocate(sizes, value=73)
        refs = await target.allocate(sizes)
        start = time.perf_counter()
        await xo.copy_to(self.buffers, refs)
        cold = time.perf_counter() - start
        assert await target.verify(73)
        for _ in range(5):
            await xo.copy_to(self.buffers, refs)
        samples = []
        for _ in range(iterations):
            start = time.perf_counter()
            await xo.copy_to(self.buffers, refs)
            samples.append(time.perf_counter() - start)
        assert await target.verify(73)
        seconds = statistics.median(samples)
        return dict(
            blocks=len(sizes),
            bytes=sum(sizes),
            cold_ms=cold * 1000,
            median_ms=seconds * 1000,
            GBps=sum(sizes) / seconds / 1e9,
            iterations=iterations,
        )


async def main(args):
    scheme = None if args.backend == "socket" else args.backend
    pool = await xo.create_actor_pool(
        "127.0.0.1",
        n_process=2,
        external_address_schemes=[None, scheme, scheme],
        subprocess_start_method="spawn",
        use_uvloop=False,
    )
    try:
        actors = [
            await xo.create_actor(
                TransferActor,
                i,
                args.array_type,
                address=pool.external_address,
                allocate_strategy=ProcessIndex(i + 1),
            )
            for i in range(2)
        ]
        for sizes in (
            [1024],
            [1024**2],
            [64 * 1024**2],
            [256 * 1024**2],
            [64 * 1024] * 64,
        ):
            for src, dst in ((0, 1), (1, 0)):
                result = await actors[src].measure(actors[dst], sizes, args.iterations)
                print(
                    json.dumps(
                        dict(
                            backend=args.backend,
                            array_type=args.array_type,
                            source_gpu=src,
                            target_gpu=dst,
                            **result,
                        )
                    ),
                    flush=True,
                )
    finally:
        await pool.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["socket", "ucx", "nixl"], required=True)
    parser.add_argument("--array-type", choices=["cupy", "torch"], default="cupy")
    parser.add_argument("--iterations", type=int, default=30)
    asyncio.run(main(parser.parse_args()))
