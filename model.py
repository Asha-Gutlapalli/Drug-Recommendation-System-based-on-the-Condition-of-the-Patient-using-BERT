import warnings

import os
import subprocess

import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from transformers import BertTokenizer, BertConfig, BertForSequenceClassification

warnings.filterwarnings("ignore")

# BERT Model
class BERT():
  def __init__(self, model_path=None, config=None):
    #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.device = torch.device("cpu")

    # load tokenizer
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # load model configuration
    if config is None:
        config = BertConfig()

    # path to save model file
    if model_path is None:
      base_dir = os.path.dirname(os.path.realpath(__file__))
      model_dir = os.path.join(base_dir, '.models')

      os.makedirs(model_dir, exist_ok=True)

      url = "https://www.dropbox.com/s/jw18aln9rmg69d6/BERT_Weights.pt?dl=0"

      model_name = os.path.split(url)[-1][:-5]
      model_path = os.path.join(model_dir, model_name)

      # download model
      if not os.path.exists(model_path):
        subprocess.call(['wget', url, '-O', model_path])

    # load pre-trained model
    self.model = BertForSequenceClassification(config)
    self.model.load_state_dict(torch.load(model_path,  map_location=self.device))

  def preprocess(self, text):
    # encode text
    input_encoded = self.tokenizer.encode_plus(
                        text,
                        add_special_tokens = True,
                        max_length = 64,
                        truncation = True,
                        pad_to_max_length = True,
                        return_attention_mask = True,
                        return_tensors = 'pt',
                   )

    # setup BERT parameters
    input_ids = input_encoded["input_ids"]
    attention_mask = input_encoded["attention_mask"]

    # prepare dataset
    pred_data = TensorDataset(input_ids, attention_mask)
    # sample dataset
    pred_sampler = SequentialSampler(pred_data)
    # prepare dataloader
    pred_dl = DataLoader(pred_data, sampler = pred_sampler, batch_size = 1)

    return pred_dl

  @torch.no_grad()
  def predict(self, pred_dl):
    for s, b in enumerate(pred_dl):
      # get batch
      b = tuple(t.to(self.device) for t in b)

      # get BERT parameters
      input_idsx, attention_maskx = b

      # predict
      outs = self.model(
          input_ids = input_idsx,
          attention_mask = attention_maskx,
          token_type_ids = None,
          )

      # predictions
      logits = outs[0]
      logits = logits.detach().cpu().numpy()
      logits = np.argmax(logits, axis=-1).item()

    return logits