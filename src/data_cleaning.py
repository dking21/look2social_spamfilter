import pandas as pd
import numpy as np

adobe = ['Acrobat','Acrobat_GU','adobe_mabhatia','adobe1234567','AdobeCare','AdobeDocCloud','AdobeExpCloud','AdobeGov','adobemax','AdobeNews','AdobePartner','adobesignstx','AdobeStarProps','AdobeUK']
onespan = ['atOneSpan','OneSpan','OneSpanSign']
docusign = ['DocuSign','DocuSignAPAC','DocuSignAPI','DocuSignIMPACT','DocuSignUK','DocuSignING']
signnow = ['signnow']
combined_lst = adobe+onespan+docusign+signnow

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

columns_to_drop = ['index','id_str','source','lang','verified',
                   'location','user_id_str','geo_enabled','user_created_at',
                   'name','user_lang','screen_name','expanded_url','url']

def feature_engineering(df):
    '''
    1. drop columns which have only one unique value
    2. drop columns which have NaN for more than half of column
    3. drop row where the language is not English
    4. drop irrelevant columns
    5. convert post created time with fourier transformation
    '''
    df2 = df
    for col in df.columns:
        if col in ['spam', 'spam_own', 'spam_corporate', 'spam_known',
                   'spam_marketing', 'spam_hijack', 'spam_bot']:
                   pass
        elif df[col].nunique() == 1:
            df2 = df2.drop(col,axis=1)
    df2.dropna(thresh=(len(df)/2), axis=1, inplace=True)
    df3 = df2[df2['lang']=='en']
    df3.drop(columns_to_drop, axis=1,inplace=True)
    df3['time_created'] = pd.to_datetime(df3['created_at']).copy()
    df3['created_at_float'] = df3['time_created'].dt.hour + df3['time_created'].dt.minute/60 + df3['time_created'].dt.second/3600
    df4 = df3.drop(['created_at','time_created'],axis=1)
    df4['time_float_sin'] = np.sin((df4['created_at_float']*2*np.pi)/24)
    df4['time_float_cos'] = np.cos((df4['created_at_float']*2*np.pi)/24)
    df4.drop('created_at_float',axis=1,inplace=True)
    return df4
