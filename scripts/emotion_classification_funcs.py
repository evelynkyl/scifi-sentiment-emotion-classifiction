from nrclex import NRCLex
import pandas as pd
import numpy as np
from collections import defaultdict
import plotly.express as px


def emo_count(text):
    """ 
    Count the total number of different emotions in the given text
    
    @param    text(list of strings)
    @return   dict: proportion of the emotions in the text
    """
    emo_count_dict = defaultdict(int)
    
    for sent in text:
        data = NRCLex(sent)
        emo_d = data.raw_emotion_scores
        for emo, score in emo_d.items():
          if emo in emo_count_dict:
              emo_count_dict[emo] += score
          else: 
              emo_count_dict[emo] = score
    return emo_count_dict


def emo_count_print(text):
    """ 
    Function for easier manual evaluation,
    Show the sentence and all the classified emotions in the given sentence
    
    @param   list of text (string): list of sentences
    """    
    for sent in text:
        data = NRCLex(sent)
        emo_d = data.affect_list
        print(sent)
        print(emo_d)


def emo_df_for_eval(text):
    """ 
    Create dataframe for predicted emotions in the given text
    Give NaN when the prediction is empty
    
    @param    text(list of strings)
    @return   dataframe
    """
    emo_final_list = []
    for sent in text:
        data = NRCLex(sent)
        emo_list = data.affect_list
        if sent not in emo_final_list:
            if len(emo_list) > 0:
                emo_final_list.append([sent, emo_list])
            else: 
                emo_final_list.append([sent, ["NaN"]])
        else:
            emo_final_list.append([sent, emo_list])
    emo_df_char = pd.DataFrame(emo_final_list, columns= ['sentence', 'emo_pred'])
    return emo_df_char


def sort_dict(ind):
    """ 
    Function to sort dictionary by value in descending order
    
    @param    dict: dictionary to be sorted
    @return   dict: sorted value of the dictionary in descending order
    """
    return sorted(ind.items(), key=lambda x: x[1], reverse=True)


def dict_to_df(in_df):
    """ 
    Convert multiclass dict to multiclass dataframe
    
    @param    dict
    @return   dataframe
    """
    emo_df = pd.DataFrame.from_dict(in_df, orient='index')
    emo_df = emo_df.reset_index()
    emo_df = emo_df.rename(columns={'index' : 'Emotion Classification' , 0: 'Emotion Count'})
    emo_df = emo_df.sort_values(by=['Emotion Count'], ascending=False)
    return emo_df


def label_classifier(label):
    """ 
    Function to label emotion to binary class, 
    for seeing the overall emotion in binary class
    
    @param    text(str)
    @return   number(int)
    """
    if label == "trust":
        return 0 # positive
    if label == "positive":
        return 0 # positive
    if label == "joy":
        return 0 # positive
    if label == "anticipation":
        return 0 # positive 
    if label == "surprise":
        return 0 # positive
    if label == "fear":
        return 1 # negative
    if label == "negative":
        return 1 # negative
    if label == "sadness":
        return 1 # negative
    if label == "anger":
        return 1 # negative
    if label == "disgust":
        return 1 # negative


def multiclass_to_binary_count(df, binarynum=0):
    """ 
    Categorize the emotion into binary (pos/neg) and count their total sum
    
    @param    dataframe
    @param    the binary number presenting either positive or negative sentiment
    @return   number(int): total number of the input sentiment
    """
    df['Emotion Classification']= df['Emotion Classification'].apply(lambda x: label_classifier(x))
    binary = df[df['Emotion Classification']==binarynum] #positve emotion
    return binary["Emotion Count"].sum().astype(np.int32)


def multi_2_binary_df(df, pos_num=27):
    """ 
    Create a dataframe of the binarized emotions and shows each of their total sum
    
    @param    df: dataframe
    @param    pos_num: the binary number presenting positive sentiment, 
              the return value of "multiclass_to_binary_count"
    @return   dataframe: a dataframe with only binary emotions and their sums in the given data
    """
    neg_num = df["Emotion Count"].sum() - pos_num
    new_df = pd.DataFrame([pos_num, neg_num], index= ["positive", "negative"], columns=["Count"])
    new_df = new_df.reset_index()
    return new_df
    
    
def visualize_df(in_df, x, y, title):
    """ 
    Plot dataframe to bar chart as visualization
    
    @param    in_df: dataframe
    @param    x: X axis of the plot
    @param    y: Y axis of the plot
    @param    title: give a title for the plot
    @return   None, but show a figure of the plot
    """
    fig = px.bar(in_df, x = x, y = y, color = y, title = title, orientation='h', width = 800, height = 400)
    fig.show()
    
