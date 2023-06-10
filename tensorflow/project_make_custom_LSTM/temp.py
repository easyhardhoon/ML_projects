import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
#------------------------------------------------------------------------------
# 1. data preprocessing
data = pd.read_csv('Hotel_reviews.csv', encoding='utf-8')
def remove_emoji(text):
    if isinstance(text, str):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # 이모티콘
                                   u"\U0001F300-\U0001F5FF"  # 기호 및 문장 부호
                                   u"\U0001F680-\U0001F6FF"  # 트랜스포트 및 심볼
                                   u"\U0001F1E0-\U0001F1FF"  # 국기 및 기호
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)
    else:
        return text
def remove_hanja(text):
    if isinstance(text, str):
        hanja_pattern = re.compile("[\u4E00-\u9FFF]+")
        return hanja_pattern.sub(r'', text)
    else:
        return text
data['review_full'] = data['review_full'].apply(remove_emoji)
data['review_full'] = data['review_full'].apply(remove_hanja)
data.dropna(subset=['review_full', 'rating_review'], inplace=True)
train_data = data[data.index % 2 == 0].reset_index(drop=True)
test_data = data[data.index % 2 == 1].reset_index(drop=True)
train_reviews = train_data['review_full']
train_ratings = train_data['rating_review']
test_reviews = test_data['review_full']
test_ratings = test_data['rating_review']
#--------------------------------------------------------------------------------
# 2. data processing for 감정분석
# Vectorize the reviews using the CountVectorizer
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_reviews)
X_test = vectorizer.transform(test_reviews)
#--------------------------------------------------------------------------------
# 3. data training
# Train a logistic regression model to predict ratings
logreg = LogisticRegression()
logreg.fit(X_train, train_ratings)
#--------------------------------------------------------------------------------
# 4. data prediction
# Predict ratings for test data
predictions = logreg.predict(X_test)
#--------------------------------------------------------------------------------
# 5. 감정분석 - 빈도수가 많았던 단어들
# Print predicted ratings and actual reviews
for review, rating in zip(test_reviews, predictions):
    print("Review:", review)
    print("Predicted Rating:", rating)
    print("-------------")

# Extract the feature names (words)
feature_names = vectorizer.get_feature_names_out()
# Get the coefficients from the logistic regression model
coefficients = logreg.coef_[0]
# Create a dataframe with words and their coefficients
coef_df = pd.DataFrame({'Word': feature_names, 'Coefficient': coefficients})
# Sort the dataframe by coefficients
coef_df = coef_df.sort_values(by='Coefficient', ascending=False)
# Visualize top N positive and negative words
top_n = 10
top_positive_words = coef_df.head(top_n)['Word']
top_negative_words = coef_df.tail(top_n)['Word']
# Plot the word frequencies
#word_freq = vectorizer.texts_to_matrix(train_reviews, mode='count').sum(axis=0).A1
word_freq = X_train.sum(axis=0).A1
word_freq_df = pd.DataFrame({'Word': feature_names, 'Frequency': word_freq})
word_freq_df = word_freq_df.sort_values(by='Frequency', ascending=False)
plt.figure(figsize=(12, 6))
plt.bar(word_freq_df.head(20)['Word'], word_freq_df.head(20)['Frequency'])
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.title('Top 20 Most Frequent Words in Hotel Reviews')
plt.show()
#--------------------------------------------------------------------------------
# 6. 감정분석 - 평가에 좋은 영향을을 주었던 단어들
# Calculate the absolute coefficients
coef_df['Abs_Coefficient'] = np.abs(coef_df['Coefficient'])
# Sort the dataframe by absolute coefficients in descending order
coef_df = coef_df.sort_values(by='Abs_Coefficient', ascending=False)
# Extract the top N influential negative words
top_influential_positive_words = coef_df[coef_df['Coefficient'] < 0].head(top_n)['Word']
# Print the top influential negative words
print("Top", top_n, "Influential Positive Words:")
for word in top_influential_positive_words:
    print("-", word)

# 7. 감정분석 - 평가에 안좋은 영향을 주었던 단어들
# Extract the top N influential positive words
top_influential_negative_words = coef_df[coef_df['Coefficient'] > 0].head(top_n)['Word']
# Print the top influential positive words
print("Top", top_n, "Influential Negative Words:")
for word in top_influential_negative_words:
    print("-", word)
#--------------------------------------------------------------------------------
#7. 시각화
plt.figure(figsize=(8, 5))
plt.bar(top_influential_positive_words, coef_df[coef_df['Coefficient'] < 0].head(top_n)['Coefficient'], color='green')
plt.xlabel('Word')
plt.ylabel('Coefficient')
plt.title('Top Influential Positive Words')
plt.xticks(rotation=45)
plt.show()
plt.figure(figsize=(8, 5))
plt.bar(top_influential_negative_words, coef_df[coef_df['Coefficient'] > 0].head(top_n)['Coefficient'], color='red')
plt.xlabel('Word')
plt.ylabel('Coefficient')
plt.title('Top Influential Negative Words')
plt.xticks(rotation=45)
plt.show()

