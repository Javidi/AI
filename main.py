import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import argparse


parser = argparse.ArgumentParser(description="Summarizer for congressional bills.")
parser.add_argument("--congress", type=int, required=False, help="Congress session number (ex. 118)", default=118)
parser.add_argument("--bill_type", type=str, required=False, help="Type of bill (ex. 'hr' for House Resolution)", default="hr")
parser.add_argument("--bill_number", type=int, required=False, help="Bill number (ex. 3342)", default=3342)
parser.add_argument("--summary_length", type=int, required=False, help="The maximum length (in tokens) to generate. (ex. 512)", default=512)
parser.add_argument("--language", type=str, required=False, help="Language to summarize in", default="english")
args = parser.parse_args()

api_key = "lcEPkwR1hbjGj7BH1dIedohAgDpoEuwkxizO154X"

def get_bill_text(congress, bill_type, bill_number):
    base_url = "https://api.congress.gov/v3/bill/"
    endpoint = f"{base_url}{congress}/{bill_type}/{bill_number}/text?api_key={api_key}"

    headers = {"Accept": "application/json"}
    response = requests.get(endpoint, headers=headers)

    if response.status_code == 200:
        data = response.json()
        text_versions = data.get("textVersions", [])

        if not text_versions:
            return "No text versions available for this bill."

        text_url = None
        for version in text_versions:
            for format in version["formats"]:
                if format["type"] == "Formatted Text":
                    text_url = format["url"]
                    break

        if not text_url:
            return "Formatted text version not found."

        text_response = requests.get(text_url)
        if text_response.status_code == 200:
            soup = BeautifulSoup(text_response.text, "html.parser")
            return soup.get_text()
        else:
            return f"Error: Unable to fetch bill (Status Code: {text_response.status_code})"

    else:
        return f"Error: Unable to fetch bill data (Status Code: {response.status_code})"

raw_text = get_bill_text(congress=args.congress, bill_type=args.bill_type, bill_number=args.bill_number)

summarizer = pipeline("text-generation", model="Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True)

def summarize_text(text):
    prompt = f"Write a short summary of the following congressional bill in {args.language} :\n\n{text}"
    result = summarizer(prompt, max_new_tokens=args.summary_length)
    return result[0]["generated_text"]

print(summarize_text(raw_text))
