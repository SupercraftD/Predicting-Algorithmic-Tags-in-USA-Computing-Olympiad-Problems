#Takes the data from everyProblem and cleans it
#Removes unnecessary info, and performs web scraping to get the actual problem statement data
#Stores results in json

import json
import requests
import cloudscraper
from bs4 import BeautifulSoup


source = "data/everyProblem.json"
dest = "data/everyProblem.json"

problems = []

with open(source,"r") as file:
    data = json.load(file)


    for prob in data:

        if "usaco.org" not in prob["url"]: continue

        clean = {
            "uniqueId":prob["uniqueId"],
            "name":prob["name"],
            "difficulty":prob["source"],
            "tags":prob["tags"],
            "url":prob["url"]
        }

        #scrape USACO
        
        scraper = cloudscraper.create_scraper()

        response = scraper.get(prob["url"])

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            
            statement = soup.find(id="probtext-text")
            clean["text"] = statement.get_text()
        else:
            print(prob["url"])
            print(response.status_code)
            print(response.headers.get("Server"))
            print(response.text[:500])
            input()
        
        problems.append(clean)


with open(dest,"w") as file:
    json.dump(problems, file, indent=4)