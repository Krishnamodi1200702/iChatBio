from dotenv import load_dotenv
import os
import requests
from instructor import patch
import openai
from pydantic import BaseModel, HttpUrl
from typing import Optional, List

# Load environment variables (API key from .env)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Patch OpenAI with Instructor
client = patch(openai.OpenAI())

# STEP 1: Ask GPT to extract a search term from user message
class SearchQuery(BaseModel):
    keyword: str

user_message = "Can you find species related to Panthera?"

search_query = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": f"Extract the main search term from this user message to query a species database: '{user_message}'"}
    ],
    response_model=SearchQuery
)

print(f"\n🔍 Search keyword extracted by GPT: {search_query.keyword}\n")

# STEP 2: Use that keyword to query Catalogue of Life (CoL)
col_url = "https://api.checklistbank.org/dataset/3LR/nameusage/search"
params = {
    "q": search_query.keyword,
    "rank": "species",
    "limit": 5
}

response = requests.get(col_url, params=params)

# STEP 3: Parse response into your Pydantic model
class NameInfo(BaseModel):
    scientificName: str
    rank: str
    link: Optional[HttpUrl]

class ColResponse(BaseModel):
    results: List[NameInfo]

results = []

if response.status_code == 200:
    data = response.json()
    for item in data["result"]:
        usage = item.get("usage", {})
        name_info = usage.get("name", {})

        try:
            result = NameInfo(
                scientificName=name_info["scientificName"],
                rank=name_info["rank"],
                link=name_info.get("link")
            )
            results.append(result)
        except Exception as e:
            print("Skipping invalid result:", e)
else:
    print("API call failed:", response.status_code)

structured_response = ColResponse(results=results)

# STEP 4: Print structured results
print("📦 Structured Catalogue of Life results:\n")
print(structured_response.model_dump_json(indent=2))