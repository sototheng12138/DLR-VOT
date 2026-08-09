# Environments

Use separate virtual environments for the DLR models and the Chronos-2 teacher. The recorded runs used Python 3.11.14 and CUDA 12.1.

For the DLR environment, install the matching PyTorch wheel first, then the remaining pinned packages:

```bash
python -m venv .venv-dlr
source .venv-dlr/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r environments/dlr-requirements.txt
```

Create the teacher environment independently:

```bash
python -m venv .venv-teacher
source .venv-teacher/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r environments/teacher-requirements.txt
```

If another CUDA version is required, follow the PyTorch installation instructions for that platform. Such a run is compatible with the code but is not the recorded software environment.
