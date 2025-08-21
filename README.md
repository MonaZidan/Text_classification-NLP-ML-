# 📧 Spam Message Classifier (Text Classification with Machine Learning)

This project is a simple but effective **spam detection** tool that uses **Natural Language Processing (NLP)** and **machine learning** to classify SMS text messages as either **Spam** or **Ham (Not Spam)**.

The core workflow includes:
- Text preprocessing using `spaCy`
- Feature extraction using `TF-IDF`
- Classification using models like `MultinomialNB`, `ComplementNB`, and `LinearSVC`
- Interactive model selection and message prediction

---

## 🧠 Models Used

- Multinomial Naive Bayes (`MultinomialNB`)
- Complement Naive Bayes (`ComplementNB`)
- Support Vector Classifier (`LinearSVC`)

---

## 🛠️ Features

- ✅ Text preprocessing with tokenization, lemmatization, and stopword removal
- ✅ Train/Test split with stratification
- ✅ Model evaluation using classification reports and confusion matrix
- ✅ Save the best model using `joblib`
- ✅ Predict new custom SMS messages

---

## 📁 Dataset

This project uses the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/team-ai/spam-text-message-classification) which contains 5,572 labeled messages.

| Column    | Description          |
|-----------|----------------------|
| `Category`| Spam or Ham label    |
| `Message` | The SMS message text |

> Make sure your CSV file is structured with these columns and paths are correct in the script.

---
