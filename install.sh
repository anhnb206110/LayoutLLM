pip install git+https://github.com/huggingface/transformers.git --quiet
pip install accelerate --quiet
pip install bitsandbytes --quiet
pip install datasets --quiet
pip install paddlepaddle-gpu --quiet
pip install "paddleocr>=2.0.1" --quiet

# download DocLocal4K
git clone https://www.modelscope.cn/datasets/iic/DocLocal4K.git
tar -xzvf DocLocal4K/imgs.tar.gz > /dev/null
rm -rf DocLocal4K