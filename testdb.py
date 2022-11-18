from pymongo import MongoClient
import json

# cluster = MongoClient("mongodb+srv://Yms98:Alphagolf212@drugs.ry1tiin.mongodb.net/?retryWrites=true&w=majority")
#
# db = cluster["Drugs"]
# collection = db["Drug"]
#
# drug_details = collection.find_one({"drug_name": "omeprazole"})
# print(drug_details['uses'])

with open('data.json') as json_file:
    drug_data = json.load(json_file)

print((dict(drug_data).get('omeprazole')['warnings']))

