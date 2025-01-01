# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "snakeviz",
# ]
# ///
import argparse
import cProfile
import os
import subprocess
import threading
import time

import psutil


def log_resource_usage(interval=1, output_file="resource_usage.log"):
    """Logs the CPU usage per core and memory usage in both percentage and MB every `interval` seconds."""
    with open(output_file, "w", encoding="utf-8") as f:

        core_columns = "\t".join(
            map(lambda i: f"Core {i}[%]", range(psutil.cpu_count()))
        )  # type: ignore
        f.write(f"Total CPU[%]\tMemory[%]\tMemory[MB]\t{core_columns}\tTime\n")
        while True:
            # Overall CPU and memory usage
            total_cpu_usage = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            memory_usage_percent = memory.percent
            memory_usage_mb = memory.used / (1024**2)  # Convert to MB

            # Per-core CPU usage
            per_core_cpu_usage = psutil.cpu_percent(percpu=True)
            per_core_cpu_str = "\t".join(map(str, per_core_cpu_usage))

            # Current timestamp
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

            # Log the data to file
            f.write(
                f"{total_cpu_usage}\t{memory_usage_percent}\t{memory_usage_mb:.2f}\t{per_core_cpu_str}\t{timestamp}\n"
            )
            f.flush()
            time.sleep(interval)


def profile_program(script, profiler_file="output.pstats"):
    """Runs the specified program with cProfile."""
    # cProfile.run(f"exec(open('{file}').read())", profiler_file)
    subprocess.run(
        ["python", "-m", "cProfile", "-o", profiler_file, script], check=True
    )


if __name__ == "__main__":
    # Set up argument parser for CLI
    parser = argparse.ArgumentParser(
        description="Profile a Python script and log resource usage."
    )
    parser.add_argument("script", help="The Python script to execute and profile.")
    parser.add_argument(
        "--profiler_file",
        default="logs/output.pstats",
        help="File to save profiler results.",
    )
    parser.add_argument(
        "--resource_file",
        default="logs/resource_usage.log",
        help="File to log CPU and memory usage.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1,
        help="Interval (in seconds) between resource usage logs.",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Start resource usage logging in a separate thread
    resource_thread = threading.Thread(
        target=log_resource_usage,
        kwargs=dict(interval=args.interval, output_file=args.resource_file),
    )
    resource_thread.daemon = True  # Daemonize thread to exit with the program
    resource_thread.start()

    # Profile the main program
    profile_program(args.script, args.profiler_file)

    print(
        f"Finalized execution to visualize results in interactive window run:\npip install uv\nuvx snakeviz {args.profiler_file}\n"
        f"To create call graph tree in svg format run:\nuvx gprof2dot -f pstats {args.profiler_file} | dot -Tsvg -o logs/profiler.svg"
    )
