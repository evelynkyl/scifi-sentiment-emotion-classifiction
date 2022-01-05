from sklearn.metrics import f1_score
import numpy as np

def clean_df_for_emoeval(indf, columns=["actual_label", "pred", "label"], axis=1):
    """ 
    Function for cleaning dataframe to remove unnecessary columns
    
    @param   indf: dataframe
    @param   columns: list of strings, names of the column to be removed
    @param   axis: apply to either column (1) or row (0) of the dataframe
    @return  cleaned dataframe
    """
    # drop columns that are not needed
    indf = indf.drop(columns, axis=1)
    # rename the existing columns so that they are the same as the prediction df
    indf.columns = ["sentence", "emo_pred"]
    return indf


def find_difference_in_df(df_true, df_pred):
     """ 
     Find the different between two dataframes 
     
     @param    df_true: dataframe with true labels (reference labels)
     @param    df_pred: dataframe with predicted labels
     @return   dataframe: a dataframe containing the differences between the dataframes
     """
     diff_list = [x for x in list(df_true['sentence'].unique()) if x not in list(df_pred['sentence'].unique())]
     dff_df = df_true[(df_true['sentence'].isin(diff_list))]
     return dff_df


def normalize_NA_pred(df_true, df_pred):
    """
    Function to normalize the difference between two dataframes
    by appending the missing sentences found in the find_diff_in_df function
    to the current prediction df to get a fully complete df 
    with all the sentences as it is in the original dataset
    
    @param    df_true: dataframe with true labels (reference labels)
    @param    df_pred: dataframe with predicted labels
    @return   dataframe: a full dataframe with all the sentences and the predictions
    """
    missing_sent_df = find_difference_in_df(df_true, df_pred)
    current_df = df_pred
    return missing_sent_df.append(current_df, ignore_index=True) 


def compute_accuracy(y_true, y_pred):
    """ 
    Function to compute the accuracy of the predictions 
    
    @param   y_true: list of true labels
    @param   y_pred: list of predictions
    @return  number(int): accuracy of the predictions
    """
    return np.sum(np.equal(y_true, y_pred)) / len(y_true)


def compute_accruacy_multilabel(true_df, pred_df):
    """ 
    Function to compute the accuracy of the predictions for multilabe classification
    
    @param   y_true: list of true labels
    @param   y_pred: list of predictions
    @return  number(int): accuracy of the predictions
    """
    true_dict = true_df.to_dict()
    pred_dict = pred_df.to_dict()
     
    accuracy = []
    
    assert len(true_dict["sentence"]) == len(pred_dict["sentence"])
    assert len(true_dict["emo_pred"]) == len(pred_dict["emo_pred"])
    
    for idx, tru_label in true_dict["emo_pred"].items():
        for i, pred_label in pred_dict["emo_pred"].items():
            if idx == i:
                interiem_accruacy = []
                for label in pred_label:
                    if label in tru_label:
                        interiem_accruacy.append(1)
                    else:
                        interiem_accruacy.append(0)
                cal_interiem_acc = sum(interiem_accruacy) / len(tru_label)
                accuracy.append(cal_interiem_acc)
                
    return np.mean(accuracy) 
