import pdfplumber
import pandas as pd
import os

# Load skills from CSV
skills_df = pd.read_csv("data/skills.csv")
skills_list = skills_df["Skill"].str.lower().tolist()

resume_folder = "data/resumes"

for file in os.listdir(resume_folder):

    if file.endswith(".pdf"):

        file_path = os.path.join(resume_folder, file)

        text = ""

        with pdfplumber.open(file_path) as pdf:

            for page in pdf.pages:

                page_text = page.extract_text()

                if page_text:
                    text += page_text

        text_lower = text.lower()

        extracted_skills = []

        for skill in skills_list:

            if skill in text_lower:
                extracted_skills.append(skill)

        print("="*60)
        print("Resume:", file)
        print("="*60)

        print("Skills Found:")

        for skill in extracted_skills:
            print("-", skill)

        print("\n")