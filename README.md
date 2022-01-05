# Sentimeny Analysis and Emotion Classification on Science Fiction (Dystopian Literature)

Using binary sentiment analysis to classify literary texts into positive/negative sentiment, and multi-class emotion classification ten emotion categories: fear, anger, anticipation, trust, surprise, positive, negative, sadness, disgust, joy. This project also performed emotion profiling for the top 3 characters in one of the books (2BR02B by Kurt Vonnegut) in the collection.

It is in fulfilment of the term project of digital philology. The goal of this project is to investigate the prominenet sentiments and emotions in this literary genre, thereby exploring the conceptualization of political and social issues in the genre of Dystopian fiction through the lens of sentiment analysis. It is an extension of my previous project of topic modeling in this genre as a means of distant reading.

This repo includes scripts for data preparation, deep learning using RoBERTa for sentiment analysis as well as traditional emotion classification classification using [NRCLex](https://github.com/metalcorebear/NRCLex), and evaluation (F1, accuracy).


## Dependencies
The library below is needed to run the scripts.
```
NRCLex
transformers
pytorch
sklearn
spacy
```
