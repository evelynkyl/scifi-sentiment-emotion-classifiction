import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

# Create class for data preparation
class SimpleDataset:
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts
    
    def __len__(self):
        return len(self.tokenized_texts["input_ids"])
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tokenized_texts.items()}
    

def count_polarity(dataframe):
    """ 
    Count the number of positive and negative labels in the dataset
    
    @param    dataframe: dataframe containing positive and negative labels
    @return   tuple: counts of the labels
    """
    dataframe = dataframe.apply(lambda x : True
                if x['label'] == "POSITIVE" else False, axis = 1)
  
    # Count number of True in the series
    positive_labels = len(dataframe[dataframe == True].index)
    negative_labels = len(dataframe) - positive_labels
    
    return positive_labels, negative_labels


def polarity_ratio(dataframe, positive_num, negative_num):
    """ 
    Count the ratio of positive and negative labels in the dataset 
    
    @param    dataframe: dataframe containing positive and negative labels
    @param    positive_num: integer, count of positive labels
    @param    negative_num: integer, count of negative labels
    @return   tuple: ratio of the pos/neg labels
    """
    return positive_num/len(dataframe), negative_num/len(dataframe)


def sentiment_analyzer(in_data):
    """ 
    Perform sentiment analysis on the given dataframe using a RoBERTa pretrained model 
    
    @param    in_data: list of strings (sentences)
    @return   dataframe with the sentences, predicted sentiment labels, and scores
    """
    # Load tokenizer and model, create trainer
    assert isinstance(in_data, list)
    in_text = in_data

    model_name = "siebert/sentiment-roberta-large-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    trainer = Trainer(model=model)

    # Tokenize texts and create prediction data set
    tokenized_texts = tokenizer(in_text,truncation=True,padding=True)
    pred_dataset = SimpleDataset(tokenized_texts)

    # Run predictions
    predictions = trainer.predict(pred_dataset)

    # Transform predictions to labels
    preds = predictions.predictions.argmax(-1)
    labels = pd.Series(preds).map(model.config.id2label)
    scores = (np.exp(predictions[0])/np.exp(predictions[0]).sum(-1,keepdims=True)).max(1)

    # Create DataFrame with texts, predictions, labels, and scores
    analyzed_df = pd.DataFrame(list(zip(in_text,preds,labels,scores)), columns=['text','pred','label','score'])
    
    return analyzed_df

