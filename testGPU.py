import platform
import sys

import torch


def bytes_to_gib(value: int) -> float:
    return value / (1024**3)


def main() -> None:
    print("=== System Information ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA: {torch.version.cuda}")
    print(f"cuDNN: {torch.backends.cudnn.version()}")
    print()

    print("=== CUDA Check ===")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Please check the NVIDIA driver "
            "and install a CUDA-enabled version of PyTorch."
        )

    device_count = torch.cuda.device_count()
    print(f"CUDA available: True")
    print(f"GPU count: {device_count}")

    for index in range(device_count):
        properties = torch.cuda.get_device_properties(index)
        print()
        print(f"GPU {index}: {properties.name}")
        print(
            f"  Compute capability: "
            f"{properties.major}.{properties.minor}"
        )
        print(
            f"  Total memory: "
            f"{bytes_to_gib(properties.total_memory):.2f} GiB"
        )
        print(f"  Multiprocessors: {properties.multi_processor_count}")

    print()
    print("=== CUDA Operation Test ===")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    try:
        x = torch.randn(
            (1024, 1024),
            device=device,
            dtype=torch.float32,
        )
        y = torch.randn(
            (1024, 1024),
            device=device,
            dtype=torch.float32,
        )
        result = x @ y

        torch.cuda.synchronize(device)

        if not torch.isfinite(result).all().item():
            raise RuntimeError(
                "CUDA operation produced NaN or Inf values."
            )

        print("CUDA matrix multiplication: PASSED")
        print(f"Result shape: {tuple(result.shape)}")
        print(f"Result dtype: {result.dtype}")
        print(f"Result device: {result.device}")
        print(
            f"Allocated memory: "
            f"{bytes_to_gib(torch.cuda.memory_allocated(device)):.3f} GiB"
        )
        print(
            f"Reserved memory: "
            f"{bytes_to_gib(torch.cuda.memory_reserved(device)):.3f} GiB"
        )

    except RuntimeError as error:
        raise RuntimeError(
            "CUDA is detected, but the CUDA operation failed. "
            "Check whether the NVIDIA driver and PyTorch build support "
            "your GPU architecture."
        ) from error

    print()
    print("GPU environment test completed successfully.")


if __name__ == "__main__":
    main()