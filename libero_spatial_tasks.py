#!/usr/bin/env python3
"""Helpers for enumerating local LIBERO-Spatial tasks."""

from __future__ import annotations

from pathlib import Path


LIBERO_SPATIAL_DATASET_DIR = Path("/home/ec2-user/LIBERO/libero/datasets/libero_spatial")
LIBERO_SPATIAL_INIT_DIR = Path("/home/ec2-user/LIBERO/libero/libero/init_files/libero_spatial")


def get_libero_spatial_tasks() -> list[dict[str, object]]:
    from libero.libero.benchmark import get_benchmark

    benchmark = get_benchmark("libero_spatial")()
    tasks: list[dict[str, object]] = []
    for task_id in range(benchmark.n_tasks):
        task = benchmark.get_task(task_id)
        task_name = task.name
        demo_hdf5 = LIBERO_SPATIAL_DATASET_DIR / f"{task_name}_demo.hdf5"
        init_path = LIBERO_SPATIAL_INIT_DIR / task.init_states_file
        tasks.append(
            {
                "task_id": task_id,
                "task_name": task_name,
                "language": task.language,
                "problem_folder": task.problem_folder,
                "init_states_file": task.init_states_file,
                "demo_hdf5": demo_hdf5,
                "init_states_path": init_path,
            }
        )
    return tasks
