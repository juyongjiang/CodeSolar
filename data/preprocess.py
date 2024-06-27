import os
import csv
from datasets import load_dataset
from utils import compute_cost


# Convert hf dataset to csv
# See more details about dataset preparation for fine-tuning at https://docs.predibase.com/user-guide/fine-tuning/prepare-data
def hfdataset_to_csv(datalist, csv_file_name, max, split):
    template = {
        "prompt":
        """<|im_start|>system\nPlease solve the following programming problem.<|im_end|>
<|im_start|>problem\n {problem}
<|im_start|>solution\n""",
        "completion": "{solution}<|im_end|>",
        "split": split}

    with open(csv_file_name, 'w', newline='') as csvfile:
        fieldnames = template.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, d in enumerate(datalist):
            if i >= max and max != -1:
                break

            row = {
                "prompt": template["prompt"].format(problem=d["problem"]), # the column name should be modified for different hf dataset
                "completion": template["completion"].format(solution=d["solution"]),
                "split": "train" # Should be either train or evaluation
            }
            writer.writerow(row)


def validate_data_csv(csv_file_name):
    """ Make sure it has `prompt`, `completion`, and `split` with all values """
    with open(csv_file_name, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            assert row['prompt']
            assert row['completion']
            assert row['split']

    return True


def load_processed_dataset(predibase, pb_dataset_name=None, hf_dataset_name=None, max=-1):
    if pb_dataset_name:
        try:
            # pb_dataset_name = "magicoder-oss-instruct-75k"
            pb_dataset = predibase.datasets.get(pb_dataset_name)
            print(f"Dataset found at Predibase: {pb_dataset}")
        except:
            print(f"Dataset not found in Predibase, please use HuggingFace dataset name!")
    elif hf_dataset_name:
        print(f"Dataset found at HuggingFace: {hf_dataset_name}")

        dataset_name = hf_dataset_name.split('/')[-1].lower() + "" if max==-1 else hf_dataset_name.split('/')[-1].lower() + f"-{max}"
        dataset_path = os.path.join("./data", dataset_name)
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
            hf_dataset = load_dataset(hf_dataset_name, trust_remote_code=True)
            train_hfdataset = hf_dataset["train"]
            csv_file_name_train = f"{dataset_path}/train.csv"
            hfdataset_to_csv(train_hfdataset, csv_file_name_train, max, "train")
            print(f"Dataset Validation: {validate_data_csv(csv_file_name_train)}")
            print(f"One step FT Cost: {compute_cost(csv_file_name_train)} USD")
            try:
                test_hfdataset = hf_dataset["test"] # it may not have test dataset
                csv_file_name_test = f"{dataset_path}/test.csv"
                hfdataset_to_csv(test_hfdataset, csv_file_name_test, max, "evaluation")
                print(f"Dataset Validation: {validate_data_csv(csv_file_name_test)}")
                print(f"One step FT Cost: {compute_cost(csv_file_name_test)} USD")
            except KeyError:
                print("No test dataset found, splitting the train dataset into train and test.")

            print("Uploading dataset to Predibase...")
            pb_dataset = predibase.datasets.from_file(csv_file_name_train, name=dataset_name)
            print("Successfully!")
        else:
            print("HuggingFace dataset already exists at local path, skipping the dataset preparation.")
            pb_dataset = predibase.datasets.get(dataset_name)
    else:
        raise ValueError("Please provide either pd dataset name or hf dataset name (or local path), and check your dataset name again.")
    
    return pb_dataset

if __name__ == "__main__":
    from predibase import Predibase
    api_token = os.environ.get("PREDIBASE_API_KEY")
    pb = Predibase(api_token=api_token)
    load_processed_dataset(pb, hf_dataset_name="/hpc2hdd/home/jjiang472/OpenSource/Datasets/Magicoder-OSS-Instruct-75K", max=100)