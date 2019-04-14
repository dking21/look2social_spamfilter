# look2social_spamfilter

Business Understanding:

Look2Social is a social media analysis platform which collects data of its client’s products and services from social media, compares with competitors and provides strategies to match emerging market trends.
However, there are numerous “spam” posts on social media nowadays. Many companies use social media as their marketing tool and some products post an automatic message on users’ social media[screenshot]. These spam posts do not convey any beneficial insight to analysis - they just increase the volume of social media posts to interpret and possibly affect some categories which are related to quantitative analysis.
The current data collection tool of Look2Social cannot filter out spam messages. Therefore, the algorithm to filter spam messages would be highly valuable as it would contribute to the improvement of accuracy and efficiency of Look2Social’s service to its clients.

Data Understanding:

Data sets were provided by look2social[screenshot]. It is an excel file that contains the data of Twitter posts related to its client and client’s competitors, which is 20,000 rows long. There are several columns such as posting date, content, uploader, hashtag and more. This data is a mixture of both spam and non-spam posts, where spam posts are not labeled. By analyzing certain pattern and eyeballing, every row in the data set got labeled.
There are four different types of spam requested by Look2Social:

Bot-generated : automatic post after using the product
Corporate posted : post written by official accounts of corporate
Marketing : post with the intention of marketing (by individual in corporate or hired marketing professionals)
Hijacking : post that uses the hashtag but it is not actually about the product

Data Cleaning:

Initially, I removed some columns - column with only one unique value, column where more than half of rows are filled with no value, column that is irrelevant to classify spam, and column that possesses repetitive information that can be found from other columns. Based on the advice from Jack Bennetto, I converted the post uploaded time with Fourier transformation so the model can use it as a predictor. Fortunately, numerical predictors did not have any missing values so I did not have to concern about it.
I created a categorical column to indicate true or false value based on the fact if user wrote a self introducing description or not. If there was no description, I converted missing value to empty string.
I wrote a function to convert column with text (targets: tweet post column and account description column).The function filters stopwords(imported from NLTK library) and punctuation, breaks the sentence down to words, lemmatizes it (imported from NLTK library) and then vectorize it with TF-IDF vectorizer (imported from scikit-learn library).

Modeling:

For efficiency, I saved final data file as 'testing2.xlsx'. Here is the final list of predictors I used for this model. For efficient explanation, I grouped it by similarity in characteristic of columns.


'Docusign', 'onespan', 'signnow','adobe sign'

These are categorical columns to indicate which company is the row related to. For example, if the tweet post is about DocuSign, the value for DocuSign becomes 1 and other columns become 0.

'listed_count', 'statuses_count','followers_count','favourites_count', 'friends_count'

These are numerical columns to indicate the activity level of user and feedback toward one's post.

'time_float_sin','time_float_cos'

These are numerical columns of post created time which was converted by Fourier transformation. Sin is the y-value, Cos is the x-value of transformed time.

'is_description_none'

This is a categorical column to indicate if account description is empty or not.

Process of data cleaning and testing is recorded at my Jupyter Notebook files (progress_diary 1~6).
I decided to test models with three types of classification models: Logistic Regression, Random Forest, and Gradient Boosting. All models were scored by AUC score (how accurately can it classify spam and non-spam) and log loss score (how accurately can it calculate the probabilities of spam).

Initially I optimized the number of predictors for best performing model. After testing from 0 text predictors to 14,000 text predictors, I found 700 predictors from tweet post text and 700 predictors from account description text were optimal value for best prediction.

By setting the number of predictors equal, I compared three different classification models. I used default setting for logistic regression (As there is a wide variety of tweet post, I think penalizing outlier can be risky), 500 estimators for random forest, and 500 estimators and depth of 2 for gradient boosting. Parameters for gradient boosting were found from experiment (estimators from 300 to 700, depths from 2 to 6) and I put equal number of estimators for random forest to see reasonable comparison.

I shuffled the data, then split the data by 80:20 for two-stage testing.
I train-test split the data to train and test the model with initial 80% of data. Train-Test ratio was 75:25. I recorded the log loss score and AUC score of initial test.
After that, I train-test split the data to train and test the model with remaining 20% of data. Train-Test ratio was 75:25. I recorded the log loss score and AUC score of final test as well.

To handle large size of predictors, I used Amazon AWS EC2 for computation.

Evaluation:

I decided to evaluate the model by predicting marketing spam, which is most ambiguous and challenging spam to predict.

Although logistic regression model showed decent log loss score, its AUC score was too low (0.502, which is almost guessing).  Random Forest model showed best log loss score and Gradient Boosting model showed best AUC score. I decided to go with Random Forest as it has an advantage of faster computation than gradient boosting (took ~3x of time for computation than RF) and the difference between two is not significant.

Using random forest with 500 estimators, I evaluated the model for different spam criteria as well.
It showed outstanding (almost perfect) classification against bot generated and corporate posted types. As they are fairly visible in terms of pattern and uniqueness, this performance was expected.
It also showed decent classification result against hijack type. It was beyond my expectation as I thought it would be challenging as marketing spam.  Just as I expected, classifying marketing spam showed under-performance. One possible reason I can think of this is the lack of consistency during labeling marketing spam - it was one of topic that I had to eyeball for spam labeling but more ambiguous compared to hijack type. It would be worth to try once again with consistently labeled data.

Each spam had different combination of important features. Based on each combination, I was able to draw important insights for Look2Social. As both text predictors and numerical/categorical predictors are included in the combinations, I concluded that using both parameters can result the great synergy in spam detection.



References

1. Galvanize Data Science Immersive Program Lectures
2. Python for Data Science and Machine Learning Bootcamp by Jose Portilla
