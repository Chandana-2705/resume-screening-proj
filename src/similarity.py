import os
import pdfplumber

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# Load AI model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Read Job Description
with open("data/job_description.txt", "r", encoding="utf-8") as file:
    job_description = file.read()

# Convert JD into embedding
jd_embedding = model.encode(job_description, convert_to_tensor=True)

resume_folder = "data/resumes"

for resume in os.listdir(resume_folder):

    if resume.endswith(".pdf"):

        resume_path = os.path.join(resume_folder, resume)

        resume_text = ""

        with pdfplumber.open(resume_path) as pdf:

            for page in pdf.pages:

                page_text = page.extract_text()

                if page_text:
                    resume_text += page_text

        # Convert Resume into embedding
        resume_embedding = model.encode(resume_text, convert_to_tensor=True)

        # Calculate similarity
        similarity = cos_sim(jd_embedding, resume_embedding)

        score = similarity.item() * 100

        print("=" * 60)
        print("Resume:", resume)
        print("Similarity Score: {:.2f}%".format(score))
        print("=" * 60)