import csv
from tokenizers import Tokenizer


def compute_cost(csv_file_name, price_per_million_tokens=0.5, tokenizer_name="upstage/solar-1-mini-tokenizer"):
    """ Compute the cost of the dataset """
    tokenizer = Tokenizer.from_pretrained(tokenizer_name)
    
    total_num_of_tokens = 0
    with open(csv_file_name, 'r') as f:
        reader = csv.DictReader(f)
        # get all values
        for row in reader:
            value = row['completion']+ " " + row['prompt']
            # tokenize
            enc = tokenizer.encode(value)
            num_of_tokens = len(enc.tokens)
            total_num_of_tokens += num_of_tokens

    return total_num_of_tokens / 1000000 * price_per_million_tokens

def download_adapter(predibase, adapter_id):
    """ Download adapter """
    predibase.adapters.download(adapter_id, dest=f"adapter_{adapter_id.split('/')[0]}.zip")
    # Unzip
    import zipfile
    with zipfile.ZipFile(f"adapter_{adapter_id.split('/')[0]}.zip", 'r') as zip_ref:
        zip_ref.extractall(f"adapter_{adapter_id.split('/')[0]}")
    
def upload_adapter(predibase, adapter_path, repo_name):
    """ Upload Adapter """
    adapter = predibase.adapters.upload(adapter_path, repo_name, "solar-1-mini-chat-240612")
    print(adapter)