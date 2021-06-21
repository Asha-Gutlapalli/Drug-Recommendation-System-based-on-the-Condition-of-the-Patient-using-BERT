import pandas as pd

import streamlit as st

from utils import *


# this caches the output to store the output and not call this function again
# and again preventing time wastage. `allow_output_mutation = True` tells the
# function to not hash the output of this function and we can get away with it
# because no arguments are passed through this.
# https://docs.streamlit.io/en/stable/api.html#streamlit.cache
@st.cache(allow_output_mutation=True, show_spinner=False)
def get_models():
  from model import BERT
  return {
    'BERT' : BERT()
  }

# load all the models before the app starts
with st.spinner('Loading Model...'):
  MODELS = get_models()

# description
st.markdown("# :syringe: **Drug Recommendation System** :pill:")
st.write('''
Patients are recommended drugs based on their condition \
and reviews of the drugs from other patients. They can \
submit their own review of the drug as well that would \
help future patients. A BERT model fine-tuned on text \
classification was employed for this objective where it \
identifies whether the reviewer recommends or does not \
recommend a particular medicine.
''')

# choose model
model = MODELS['BERT']

# navigation
st.sidebar.markdown("## **Navigation**")
navigation = st.sidebar.radio("Go to:", ["Home", "Patient Reviews", "Submit Feedback"])

# conditions
conditions = get_conditions()

# page definitions
if navigation == "Home":
  # instruction
  st.markdown("### :mega: **Get your drug recommendations!** :mega:")
  # select condition
  condition = st.selectbox(label="What is your condition?", options=conditions)

  if condition:
    # check if recommended drugs exist
    try:
      # get drugs
      fig, drugs = get_drugs(condition)
      # display pie chart
      st.plotly_chart(fig, use_container_width=True)
      # display
      st.table(drugs)
    except:
      st.info("Sorry, no recommended drugs exist as of now!")
elif navigation == "Patient Reviews":
  # instruction
  st.markdown("### :open_file_folder: **Read reviews by fellow patients!** :open_file_folder:")
  # select condition
  condition = st.selectbox(label="Condition", options=conditions)

  if condition:
    # get drugs
    _, drugs = get_drugs(condition)
    # select drug
    drug = st.selectbox(label="Drugs", options=drugs)

    if drug:
      # get reviews
      info = get_info(condition, drug)
      # number of reviews
      num_reviews = st.slider('Number of Reviews', 0, len(info), 1)
      # display reviews in a container
      for i in range(num_reviews):
        container(info[i])
elif navigation == "Submit Feedback":
  # instruction
  st.markdown("### :heavy_check_mark: **Please share your feedback!** :heavy_check_mark:")
  # form
  with st.form("Feedback"):
  # select condition
    condition = st.selectbox(label="Condition", options=conditions)

    if condition:
      # get drugs
      _, drugs = get_drugs(condition)
      # select drug
    drug = st.selectbox(label="Drug", options=drugs)
    # review
    default_ = ""
    review = st.text_area("Review", value=default_, key="Text")
    # get date
    date = get_date()
    # submit form
    submitted = st.form_submit_button("Submit")

    if submitted:
      # validate form
      if not review:
        st.warning('Please fill all fields!')
      else:
        # predict recommendation
        # get dataloader
        pred_dl = model.preprocess(review)
        # get prediction
        recommend = model.predict(pred_dl)

        # insert in db
        insert(drug, condition, review, recommend, date)

        # successful submission
        st.success("Successfully submitted!")