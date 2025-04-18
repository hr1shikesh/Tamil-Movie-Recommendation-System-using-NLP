# 🎬 Tamil Movie Recommendation System using NLP

This project builds a content-based movie recommender system leveraging Natural Language Processing (NLP) techniques, trained on Tamil movie reviews. By using TF-IDF and cosine similarity, the model is able to recommend similar movies based on textual content.

---

## 📂 Project Structure

- `tamil_movie_reviews_train.csv`: Training data with Tamil movie reviews.
- `tamil_movie_reviews_test.csv`: Test data to evaluate recommendations.
- `processed_train_data.csv`: Cleaned and processed version of the training data.
- `movie.csv` / `with_movie_names.csv`: Extended movie metadata for training/evaluation.
- `tfidf_vectorizer.joblib`, `tfidf_matrix.joblib`: Serialized vectorizer and matrix.
- `train_similarity.joblib`: Cosine similarity matrix.
- `final_model.ipynb`, `workingModel.ipynb`, `test.ipynb`: Jupyter notebooks containing the model creation, training, and evaluation steps.
- `maybe.py`, `temp.py`, `workingmodel.py`: Python scripts for deployment or experimentation.

---

## 🧠 Technologies Used

- Python 3
- Pandas, NumPy
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity
- Jupyter Notebook
- Joblib (for model serialization)

---

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/NLP-Movie-Recommender.git
   cd NLP-Movie-Recommender
