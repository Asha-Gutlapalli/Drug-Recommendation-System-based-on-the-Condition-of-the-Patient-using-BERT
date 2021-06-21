# Drug Recommendation System based on the Condition of the Patient using Bidirectional Encoder Representations from Transformers

Patients are recommended drugs based on their condition and reviews of the drugs from other patients. They can submit their own review of the drug as well that would help future patients. A BERT model fine-tuned on text classification was employed for this objective where it identifies whether the reviewer recommends or does not recommend a particular medicine.


## BERT

[BERT](https://arxiv.org/pdf/1810.04805.pdf) is the Bidirectional Encoder Representations from Transformers developed by Google. It is a attention mechanism that unlike a regular transformer consists only of an encoder network. It learns the correlation between words by ingesting the entire sequence of tokens at once. BERT is pre-trained on two NLP tasks namely Masked Langauge Modelling (MLM) and Next Sentence Prediction (NSP). In simple words, MLM facilitates BERT to understand the relationship between words while NSP facilitates BERT to understand the relationship between sentences. BERT can be used to finetune on a wide range of NLP applications such as classification, Question Answering, and Named Entity Recognition to name a few.

<p align="center">
  <img src="/assets/Pre-training.png">
  <img src="/assets/Fine-Tuning.png">
</p>


## Install Packages

Install required libraries using the following command:
```bash
$ pip install -r requirements.txt
```


## Files

- `/notebooks`: Notebooks on BERT and MongoDB
- `/assets`: All images and GIFs displayed on this README.md
- `utils.py`: Consists of helper functions for MongoDB and Streamlit
- `model.py`: BERT model
- `run.py`: Streamlit app


## Train, Evaluate, and Test

Check out the notebook [BERT.ipynb](/notebooks/BERT.ipynb) to train, evaluate, and test your own model on custom datasets.


## MongoDB

### Follow the steps below to set up your own collection using MongoDB:

- [Install MongoDB](https://docs.mongodb.com/guides/server/install/)
- [Secure your MongoDB Deployment](https://docs.mongodb.com/guides/server/auth/)
- [Connect to MongoDB](https://docs.mongodb.com/guides/server/drivers/)
- Refer this [notebook](/notebooks/MongoDB.ipynb) to create the collection.

### Connect MongoDB to Streamlit

Read this [tutorial](https://docs.streamlit.io/en/0.83.0/tutorial/mongodb.html) to connect MongoDB to Streamlit.


## Run Streamlit App

Use the following command to run Streamlit App:
```bash
$ streamlit run run.py

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.4:8501

```

Check out the GIFs below for demonstrations!

### Home

<img src="/assets/home.gif">

### Patient Reviews

<img src="/assets/reviews.gif">

### Submit Feedback

<img src="/assets/feedback.gif">