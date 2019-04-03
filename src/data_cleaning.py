import pandas as pd
import numpy as np

adobe = ['Acrobat','Acrobat_GU','adobe_mabhatia','adobe1234567','AdobeCare','AdobeDocCloud','AdobeExpCloud','AdobeGov','adobemax','AdobeNews','AdobePartner','adobesignstx','AdobeStarProps','AdobeUK']
onespan = ['atOneSpan','OneSpan','OneSpanSign']
docusign = ['DocuSign','DocuSignAPAC','DocuSignAPI','DocuSignIMPACT','DocuSignUK','DocuSignING']
signnow = ['signnow']
combined_lst = adobe+onespan+docusign+signnow

def labeled_data_combining(data1, data2):
    hand_partone = pd.read_excel(data1)
    hand_parttwo = pd.read_excel(data2)
    hand_combined = pd.concat([hand_partone,hand_parttwo])
    hand_combined.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1,inplace=True)
    return hand_combined

def data_labeling(data,data1,data2):
    '''
    0. combine two labeled subsets of data1&2
    1. label spam_corporate based on the user account
    2. label spam_bot based on expanded_url
    3. split data by labeled and unlabeled
    4. label spam_marketing and spam_hijack based on step0's result
    5. combine two subsets, sort it by index
    '''
    hand_partone = pd.read_excel(data1)
    hand_parttwo = pd.read_excel(data2)
    hand_combined = pd.concat([hand_partone,hand_parttwo])
    hand_combined.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1,inplace=True)
    df = pd.read_excel(data)
    df['spam_corporate'][df.screen_name.isin(combined_lst)] = 1
    df['spam_bot'][df.expanded_url == 'https://acrobat.adobe.com'] = 1
    hand_labeled_df = df[(df['spam_bot'] != 1) & (df['spam_corporate'] != 1)]
    auto_labeled_df = df[(df['spam_bot'] == 1) | (df['spam_corporate'] == 1)]
    hand_labeled_df['spam_marketing'] = hand_combined['spam_marketing'].values
    hand_labeled_df['spam_hijack'] = hand_combined['spam_hijack'].values
    combined_labeled_df = pd.concat([hand_labeled_df,auto_labeled_df])
    combined_labeled_df.sort_values(by='index', inplace=True)
    return combined_labeled_df
    
