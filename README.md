# Sentiment Analysis and Emotion Classification on Science Fiction (Dystopian Literature)

Using binary sentiment analysis to classify literary texts into positive/negative sentiment, and multi-class emotion classification ten emotion categories: fear, anger, anticipation, trust, surprise, positive, negative, sadness, disgust, joy. This project also performed emotion profiling for the top 3 characters in one of the books (2BR02B by Kurt Vonnegut) in the collection.

It is in fulfilment of the term project of Current Topics in Digital Philology. The goal of this project is to investigate the prominenet sentiments and emotions in this literary genre, thereby exploring the conceptualization of political and social issues in the genre of Dystopian fiction through the lens of sentiment analysis. It is an extension of my previous project of topic modeling in this genre as a means of distant reading.

This repo includes scripts for data preparation, deep learning using RoBERTa for sentiment analysis as well as traditional emotion classification classification using [NRCLex](https://github.com/metalcorebear/NRCLex), and evaluation (F1, accuracy).


## Dependencies
The libraries/frameworks below are needed to run the scripts.
```
NRCLex
transformers
pytorch
sklearn
spacy
```

## Dataset of this project

| Dataset |     Size    | Source       | Genre | Time period         |
| -------- | ------------  | ----------- | ------ | ----------- |
| dystro |      10 books (Details can be found in [data](https://github.com/evelynkyl/scifi-sentiment-emotion-classifiction/data)) | Project Gutenburg   | Dystopia | 19-20th century   |

## Result
The detailed results and experiments can be found in the notebook in the root directory of this repo.
### The entire collection and its subset (2BR02B)
![image](https://user-images.githubusercontent.com/40916491/149620871-1b71231c-9d15-497c-8a2e-c276c9ef0cd8.png)
![image](https://user-images.githubusercontent.com/40916491/149620966-077abf03-9b27-465c-80ad-e3a43ef33f5d.png)
![image](https://user-images.githubusercontent.com/40916491/149620958-c730d16a-c5f4-4fad-bcc4-2fc2bc906548.png)
### Characters in the subset (2BR02B)
![image](https://user-images.githubusercontent.com/40916491/149620985-cd964f21-e7ab-4b2b-8ca3-07d97389e2d2.png)
![image](https://user-images.githubusercontent.com/40916491/149620988-b169ff79-9013-416d-8a45-a43c65af42f0.png)

## Evaluation (of the characters)
![image](https://user-images.githubusercontent.com/40916491/149620994-9482cce1-73b1-4e39-9cb0-d4184b1436ee.png)
