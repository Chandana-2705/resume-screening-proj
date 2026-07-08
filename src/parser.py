import pdfplumber
import os

# Folder where resumes are stored
resume_folder = "data/resumes"

# Read every PDF in the folder
for file in os.listdir(resume_folder):
    if file.endswith(".pdf"):
        file_path = os.path.join(resume_folder, file)

        print("=" * 60)
        print("Resume:", file)
        print("=" * 60)

        with pdfplumber.open(file_path) as pdf:
            text = ""

            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text

        print(text[:1000])   # Print first 1000 characters
        print("\n")