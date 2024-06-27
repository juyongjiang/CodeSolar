# CodeSolar

![# CodeSolar](assets/codesolar.jpg)

## Installation
First, we have to install all the libraries listed in `requirements.txt`.

```bash
pip install -r requirements.txt
# export your Predibase token, found at: https://app.predibase.com/
export PREDIBASE_API_KEY=<xxxxxxxx>
```

<!-- ## Dataset Preparation

1. Download `Magicoder-OSS-Instruct-75K` instruction tuning dataset for coding 
    * locally saved
        ```
        cd data
        bash download.sh 
        hf_dataset_name = "local_data_path"
        ```
    * saved in HuggingFace `.cache`
        ```
        hf_dataset_name = "ise-uiuc/Magicoder-OSS-Instruct-75K"
        ```

2. Processing with instruction template
```
python data/preprocess.py
``` -->
## Dataset Preparation

You may need to modify the instruction template [`./data/preprocess.py`](./data/preprocess.py) for different dataset.

```python
template = {
    "prompt":
    """<|im_start|>system\nPlease solve the following programming problem.<|im_end|>
<|im_start|>problem\n {problem}
<|im_start|>solution\n""",
    "completion": "{solution}<|im_end|>",
    "split": split}
```


## Fine-tuning
* If dataset exists in [Predibase](https://app.predibase.com/data):

```bash
python finetune.py \
        --base_model_name="solar-1-mini-chat-240612" \
        --pb_dataset_name="magicoder-oss-instruct-75k" \
        --repo_name="codesolar-v1-adapter" \
        --epoch=1 \
        --learning_rate=0.0001 \
        --rank=8
```

For debugging, please use `magicoder-oss-instruct-75k-100`, only 100 examples will be used. 

* If dataset exists in [HuggingFace](https://huggingface.co/datasets):

```bash
python finetune.py \
        --base_model_name="solar-1-mini-chat-240612" \
        --hf_dataset_name="ise-uiuc/Magicoder-OSS-Instruct-75K" \
        --repo_name="codesolar-v1-adapter" \
        --epoch=1 \
        --learning_rate=0.0001 \
        --rank=8
```


## Inference
We use the `Solar-as-a-Judge`, so you need to export your `UPSTAGE_API_KEY`.

```bash
export UPSTAGE_API_KEY=<xxxxxxxx>
python inference.py
```
