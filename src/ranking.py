import os
import pdfplumber
import pandas as pd
import joblib

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# -----------------------------
# Load AI Model
# -----------------------------
nlp_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load ML Model
ml_model = joblib.load("models/candidate_model.pkl")

# -----------------------------
# Read Job Description
# -----------------------------
with open("data/job_description.txt", "r", encoding="utf-8") as file:
    job_description = file.read()

jd_embedding = nlp_model.encode(job_description, convert_to_tensor=True)

# -----------------------------
# Skills List
# -----------------------------
skills = [
    "python","java","html","css","javascript",
    "sql","mysql","flask","git","docker"
]

results = []

resume_folder = "data/resumes"

for resume in os.listdir(resume_folder):

    if resume.endswith(".pdf"):

        resume_path = os.path.join(resume_folder, resume)

        text = ""

        with pdfplumber.open(resume_path) as pdf:

            for page in pdf.pages:

                page_text = page.extract_text()

                if page_text:
                    text += page_text.lower()

        # -----------------------------
        # Similarity
        # -----------------------------

        resume_embedding = nlp_model.encode(text, convert_to_tensor=True)

        similarity = cos_sim(jd_embedding, resume_embedding).item() * 100

        # -----------------------------
        # Skills
        # -----------------------------

        feature_vector = []

        for skill in skills:

            if skill in text:
                feature_vector.append(1)
            else:
                feature_vector.append(0)

        # Experience (dummy value for now)
        experience = 2

        feature_vector.append(experience)
        feature_vector.append(similarity)

        # Predict score
        predicted_score = ml_model.predict([feature_vector])[0]

        results.append({
            "Resume": resume,
            "Similarity": similarity,
            "Predicted Score": predicted_score
        })

# -----------------------------
# Ranking
# -----------------------------

results = sorted(results, key=lambda x: x["Predicted Score"], reverse=True)

print("\n")
print("="*70)
print("FINAL CANDIDATE RANKING")
print("="*70)

for i, candidate in enumerate(results, start=1):

    print(f"\nRank {i}")

    print("Resume:", candidate["Resume"])

    print("Similarity: {:.2f}%".format(candidate["Similarity"]))

    print("ML Score: {:.2f}".format(candidate["Predicted Score"]))

print("\n")