import os
import csv
from predibase import Predibase

# Get a KEY from https://app.predibase.com/
# You can generate an API token on the homepage or find an existing key under Settings > My Profile
api_token = os.environ.get("PREDIBASE_API_KEY")
pb = Predibase(api_token=api_token)


# Get adapter by id
adapter_id = 'news-summarizer-model/29'
adapter = pb.adapters.get(adapter_id)
print(adapter)

# The examples in the training dataset should be similar to the requests at inference time
input_prompt="""
<|im_start|>system\nThe following passage is content from a news report. Please summarize this passage in one sentence or less.<|im_end|>
<|im_start|>passage
Artificial intelligence startup Upstage secured 100 billion won ($72 million)
from investors including SK Networks Co. and KT Corp. to bankroll an expansion in the US, Japan and Southeast Asia.
The South Korean startup has closed a Series B financing,
tripling the amount raised in a previous funding round in 2021,
the company said on Tuesday. New backers including Korea Development Bank and Shinhan Venture Investment Co.
as well as existing investors such as SBVA, formerly known as SoftBank Ventures Asia, took part in the latest financing.<|im_end|>
<|im_start|>summary
"""

base_model_name = "solar-1-mini-chat-240612"
lorax_client = pb.deployments.client(base_model_name)
print(lorax_client.generate(input_prompt, adapter_id=adapter_id, max_new_tokens=1000).generated_text)


# evaluation on test dataset with LLM-as-a-Judge
import solar_as_judge as saj
# os.environ.get("UPSTAGE_API_KEY")

test_csv_file_name = "tldr_news_toy/test.csv"
win_results = {"A_wins":0, "B_wins": 0, "tie": 0,  "A_score": 0, "B_score": 0}
with open(test_csv_file_name, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        prompt = row['prompt']
        ground_truth = row['completion']
        # base model
        A_answer = lorax_client.generate(prompt, max_new_tokens=1000).generated_text
        # fine-tuned model
        B_answer = lorax_client.generate(prompt, adapter_id=adapter_id, max_new_tokens=1000).generated_text

        A_score, B_score = saj.judge(prompt, A_answer, B_answer, ground_truth)
        print(A_score, B_score, A_answer, B_answer)
        
        win_results["A_score"] += A_score
        win_results["B_score"] += B_score
        if A_score > B_score:
            win_results["A_wins"] += 1
        elif B_score > A_score:
            win_results["B_wins"] += 1
        else:
            win_results["tie"] += 1
        print(win_results)
    
    
# visualize the results   
import matplotlib.pyplot as plt

# Sample data similar to the image
categories = ["Win rate", "Win score"]

win_rate_sum = win_results["A_wins"] + win_results["B_wins"]
win_score_sum = win_results["A_score"] + win_results["B_score"]
percentages1 = [round(win_results["A_wins"]*100/win_rate_sum), round(win_results["A_score"]*100/win_score_sum)]
percentages2 = [round(win_results["B_wins"]*100/win_rate_sum), round(win_results["B_score"]*100/win_score_sum)]

values1 = [win_results["A_wins"], win_results["A_score"]]
values2 = [win_results["B_wins"], win_results["B_score"]]

# Create a bar plot
# Bar positions
bar_width = 0.5
y_pos = range(len(categories))

# Plotting the bars
fig, ax = plt.subplots()
ax.barh(y_pos, percentages1, color='steelblue', edgecolor='black', height=bar_width)
ax.barh(y_pos, percentages2, left=percentages1, color='orange', edgecolor='black', height=bar_width)

# Adding text
for i in range(len(categories)):
    ax.text(percentages1[i]/2, i, f"Base {values1[i]} ({percentages1[i]}%)", ha='center', va='center', color='black')
    ax.text(percentages1[i] + percentages1[i+1]/2, i, f"Fine-tuned {values2[i]} ({percentages2[i]}%)", ha='center', va='center', color='black')

# Labels and Titles
ax.set_yticks(y_pos)
ax.set_yticklabels(categories)
ax.set_xlim(0, 100)
ax.set_xlabel('Percentage')
ax.set_title('Base VS Fine-tuned Win Score & Rate')
plt.show()