
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~ SYNTHEA ~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# https://github.com/synthetichealth/synthea
import pandas as pd
import os
import pymongo
import json


# synthea_dir = "../synthea/output/csv"
# os.listdir(synthea_dir)
# patients = pd.read_csv( os.path.join(synthea_dir,"patients.csv") )

synthea_dir = "../synthea/output/fhir"
synthea_records = [os.path.join(synthea_dir, x) for x in os.listdir(synthea_dir) if x !='.DS_Store']

myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["synthea"]
mycol = mydb["NYC_patients"]

def import_json(path):
    print("Importing..."+path)
    with open(path) as j:
         return json.load(j)

records = [import_json(x) for x in synthea_records]
x = mycol.insert_many(records)
x.inserted_id



# def import_synthea_record(record_path):
#     with open(record_path) as f:
#         record = json.load(f)
#         # soup = BeautifulSoup(f, "html.parser",from_encoding='utf-8') #'lxml'
#     return record
# record = import_synthea_record(synthea_records[0])
# def return_conditions(record):
#     conditions = {}
#     for x in record["entry"]:
#         if x["resource"]["resourceType"]=="Condition":
#             conditions = x["resource"]
#     return pd.DataFrame(conditions)
#


