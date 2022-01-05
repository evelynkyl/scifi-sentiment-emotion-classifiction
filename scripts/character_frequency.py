import spacy
import pandas as pd

nlp = spacy.load('en_core_web_sm')

def list_name_entities(text):
    """
    Count the occurence freqency of named entities in the dictionary
    This serves as a basis for further character count, 
    to take a glimpse at the varied names for each named entity (person) in the dataset
    
    @param    text: list of strings (sentences)
    @return   dataframe: count of named entites (characters) in the given text
    """
    unique_name = set()
    name_entities = []
    for sentence in text:
        doc = nlp(sentence)
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                unique_name.add(ent.text)
                name_entities.append(ent.text)
    name_counts = pd.value_counts(name_entities)
    return name_counts


def count_characters(text, in_dict): 
    """
    Count the occurence freqency of the character in the dictionary
    
    @param    txt: list of strings (tokens)
    @param    in_dict: dictionary of character names
    @return   dict: count of characters in the given text as dictionary
    """
    count_dict = {}
    for word in text:
        lower_word = word.lower()
        for name, var in in_dict.items():
            if lower_word in var:    
                if name in count_dict:
                    count_dict[name] += 1
                else:
                    count_dict[name] = 0
    return count_dict
    
    
def get_character_dialogue(book, character_dict, character):
    """
    Get the dialog of a specific character for the given book
    
    @param    book: list of strings, a book containing sentences as strings
    @param    character_dict: dictionary with character name as key and their name variations as values
    @param    character: string, the name of the character (key of the above dictionary)
    @return   list: list of sentences containing the name of the character
    """
    character_name_variations = character_dict[character]
        
    lowercase_book = [x.lower() for x in book]
    dialog_list = []
    
    for sent in lowercase_book:
        chara_dialog = [sent for name in character_name_variations if name in sent]
        
        if len(chara_dialog) < 2 and len(chara_dialog) > 0:
            dialog_list.append(chara_dialog)
        if len(chara_dialog) > 1:
            dialog_list.append([chara_dialog[0]])
            
    return dialog_list