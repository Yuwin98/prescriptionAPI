from pymongo import MongoClient

cluster = MongoClient("mongodb+srv://Yms98:Alphagolf212@drugs.ry1tiin.mongodb.net/?retryWrites=true&w=majority")

db = cluster["Drugs"]
collection = db["Drug"]

drug_details = collection.find_one({"drug_name": "omeprazole"})
print(drug_details['uses'])
