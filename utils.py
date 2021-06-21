from datetime import datetime
import re

import numpy as np
import pandas as pd

import plotly.express as px

import pymongo
import streamlit as st

#-------------------------------------------------------------------------------------------------------------------------------------------------

# Helper Functions

# get date
def get_date():
  now = datetime.now()
  date = now.strftime("%Y/%m/%d")

  return date

#-------------------------------------------------------------------------------------------------------------------------------------------------

# MongoDB

# initialize connection with MongoDB
client = pymongo.MongoClient(**st.secrets["mongo"])

# database
db = client.admin

# collection
collection = db.drugs

# get conditions
@st.cache(ttl=600)
def get_conditions():
  conditions = collection.distinct("Condition",
                                  { "Condition": { "$not": { "$regex": "<.*?>" },
                                  "$type": "string"}})

  return conditions

# get drugs for a condition
@st.cache(ttl=600)
def get_drugs(condition):
  drugs = pd.DataFrame(list(collection.aggregate([
                      {"$match" : {"Condition" : condition, "Recommend" : 1}},
                      {"$project" : {"_id" : 0, "DrugName" :  1}},
                      {"$group" : {"_id":"$DrugName", "Recommend":{"$sum":1}}},
                     ])))
  drugs.columns = ["Drug Name", "Number of Recommendations"]
  plot = px.pie(drugs, values = "Number of Recommendations", names = "Drug Name", title = f"Drug Recommendations - {condition}")

  return plot, drugs["Drug Name"]

# get info about drugs
@st.cache(ttl=600)
def get_info(condition, drug):
  cursor = list(collection.find({"Condition" : condition, "DrugName": drug},
                           { "_id" : 0, "Date" : 1, "Review" : 1, "Recommend" : 1}).sort("Date", -1))
  return cursor

# insert in db
def insert(drug, condition, review, recommend, date):
  collection.insert_one({"DrugName": drug, "Condition": condition, "Review": review, "Recommend": recommend, "Date": date})

#-------------------------------------------------------------------------------------------------------------------------------------------------

# Streamlit

# streamlit container
def container(row):
  # classes
  classes = {0: "No", 1: "Yes"}

  with st.beta_container():
    st.markdown("---")
    # date
    st.markdown("**Date:**")
    st.markdown( row["Date"])
    # review
    st.markdown("**Review:**")
    st.markdown(row["Review"])
    # recommend
    st.markdown("**Recommend:**")
    st.markdown(classes[row["Recommend"]])

#-------------------------------------------------------------------------------------------------------------------------------------------------