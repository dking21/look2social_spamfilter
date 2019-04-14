# look2social_spamfilter

Business Understanding:

Look2Social is a social media analysis platform which collects data of its client’s products and services from social media, compares with competitors and provides strategies to match emerging market trends.
However, there are numerous “spam” posts on social media nowadays. Many companies use social media as their marketing tool and some products post an automatic message on users’ social media[screenshot]. These spam posts do not convey any beneficial insight to analysis - they just increase the volume of social media posts to interpret and possibly affect some categories which are related to quantitative analysis.
The current data collection tool of Look2Social cannot filter out spam messages. Therefore, the algorithm to filter spam messages would be highly valuable as it would contribute to the improvement of accuracy and efficiency of Look2Social’s service to its clients.

Data Understanding:

Data sets were provided by look2social[screenshot]. It is an excel file that contains the data of Twitter posts related to its client and client’s competitors, which is 20,000 rows long. There are several columns such as posting date, content, uploader, hashtag and more. This data is a mixture of both spam and non-spam posts, where spam posts are not labeled. By analyzing certain pattern and eyeballing, every row in the data set got labeled.

Data Cleaning:

Initially, I removed some columns - column with only one unique value, column where more than half of rows are filled with no value, column that is irrelevant to classify spam, and column that possesses repetitive information that can be found from other columns. Based on the advice from Jack Bennetto, I converted the post uploaded time with Fourier transformation so the model can use it as a predictor. Fortunately, numerical predictors did not have any missing values so I did not have to concern about it.
I created a categorical column to indicate true or false value based on the fact if user wrote a self introducing description or not. If there was no description, I converted missing value to empty string.
I wrote a function to convert column with text (targets: tweet post column and account description column).The function filters stopwords(imported from NLTK library) and punctuation, breaks the sentence down to words, lemmatizes it (imported from NLTK library) and then vectorize it with TF-IDF vectorizer (imported from scikit-learn library).

Modeling:

Here is the final list of predictors I used for this model. For efficient explanation, I grouped it by similarity in characteristic of columns.


'Docusign', 'onespan', 'signnow','adobe sign'

These are categorical columns to indicate which company is the row related to. For example, if the tweet post is about DocuSign, the value for DocuSign becomes 1 and other columns become 0.

'listed_count', 'statuses_count','followers_count','favourites_count', 'friends_count'

These are numerical columns to indicate the activity level of user and feedback toward one's post.

'time_float_sin','time_float_cos'

These are numerical columns of post created time which was converted by Fourier transformation. Sin is the y-value, Cos is the x-value of transformed time.

'is_description_none'

This is a categorical column to indicate if account description is empty or not.



References

1. Galvanize Data Science Immersive Program Lectures
2. Python for Data Science and Machine Learning Bootcamp by Jose Portilla
