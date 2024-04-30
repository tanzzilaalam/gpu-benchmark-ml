# gpu-benchmark-ml
To check temperature and power limit failure


## Install virtual enviroment

```bash
python3.10 -m venv venv
source venv/bin/activate
pip3 install torch torchvision
```

## Run in gpu

Set GPU:

```bash
export CUDA_VISIBLE_DEVICES=0
```

Run:

```bash
python script.py \
    --batch_size 4 \
    --epochs 5 \
    --device "cuda"
```