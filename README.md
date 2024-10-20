
## Setup

Get the latest [PyTorch](https://pytorch.org/get-started/locally/)-compatible [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive).

### Windows

```ps1
# Enable Developer Mode in Windows settings
scoop install sudo innounp
sudo Set-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name 'LongPathsEnabled' -Value 1
scoop bucket add extras
scoop install mambaforge rust
mamba update mamba --all
mamba create -n signalbot python=3.12 poetry
```
