# Install packages and download data

```bash
./install.sh
```

# Prepare data

HF_TOKEN to access meta-llama2 model.

```bash
python data_prepare.py --hf_token [HF_TOKEN]
```

# Train

```bash
python train.py --hf_token [HF_TOKEN]
```