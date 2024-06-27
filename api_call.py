# CURL test
import os
import requests
import json


url = "https://serving.app.predibase.com/7ea6d0/deployments/v2/llms/solar-1-mini-chat-240612/generate"

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

adapter_id = 'news-summarizer-model/29'
api_token = os.environ.get("PREDIBASE_API_KEY")

payload = {
    "inputs": input_prompt,
    "parameters": {
        "adapter_id": adapter_id,
        "adapter_source": "pbase",
        "max_new_tokens": 512,
        "temperature": 0.8
    }
}

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_token}"
}

response = requests.post(url, data=json.dumps(payload), headers=headers)

print(response.text)