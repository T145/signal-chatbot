## Tech Stack

- Ollama to host and manage the AIs
- Prompt control defined by LangChain & LangGraph
- Weviate as the dataset vectorstore between Ollama and LangGraph
- Chat storage handled by MongoDB

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

## Runtime

```
poetry install
docker compose up -d
docker exec -it ollama ollama pull llama3.1:8b
chatbot run
```
