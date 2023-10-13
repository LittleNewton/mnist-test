# Hello AI

This is my first try to AI with the assistance of ChatGPT4.

You can use this command to check if your system support GPU-CUDA acceleration.

``` bash
docker run --gpus all --rm -v $(pwd):/workspace -w /workspace pytorch/pytorch:latest python test-cuda.py
```

You can use this command to run this project.

``` bash
docker run --gpus all --rm -v $(pwd):/workspace -w /workspace pytorch/pytorch:latest python train.py
```
