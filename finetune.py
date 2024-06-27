import os
import fire
from datasets import load_dataset
from data.preprocess import load_processed_dataset
from predibase import Predibase, FinetuningConfig


# Get a KEY from https://app.predibase.com/
# You can generate an API token on the homepage or find an existing key under Settings > My Profile
api_token = os.environ.get("PREDIBASE_API_KEY")
pb = Predibase(api_token=api_token)


# Fine-tune LLMs on coding dataset
def pb_finetune(
    base_model_name="solar-1-mini-chat-240612", 
    pb_dataset_name=None, 
    hf_dataset_name="ise-uiuc/Magicoder-OSS-Instruct-75K", 
    repo_name="codesolar-v1-adapter"
):
    # Load the dataset
    if pb_dataset_name:
        pb_dataset = load_processed_dataset(pb, pb_dataset_name) # only 100 examples for testing while magicoder-oss-instruct-75k with all
    elif hf_dataset_name:
        pb_dataset = load_processed_dataset(pb, hf_dataset_name)
    
    # Create an adapter repository
    repo = pb.repos.create(name=repo_name, description=f"Create {repo_name} repository...", exists_ok=True)
    print("repo info: ", repo)

    # Start a fine-tuning job, blocks until training is finished
    adapter = pb.adapters.create(
        config=FinetuningConfig(
            base_model=base_model_name,
            epochs=1, # default: 3
            learning_rate=0.0001, # default: 0.0002
            # if employing LoRA fine-tuning, set the following parameters (Check!)
            rank=1, # default: 16
            # target_modules=["q_proj", "v_proj", "k_proj"], # default: None (infers [q_proj, v_proj] for mistral-7b)
        ),
        dataset=pb_dataset, # Also accepts the dataset name as a string
        repo=repo, # after finish fine-tuning, the adapter weight will be pushed to this repo
        description=f"Start fine-tuning {repo_name}...",
    )
    adapter_id = adapter.repo + "/" + str(adapter.tag)
    print("adapter_id: ", adapter_id)
    print("Successfully!")


if __name__ == "__main__":
    fire.Fire(pb_finetune)