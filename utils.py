from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


#.\env\Scripts\activate
#os.environ["GROQ_API_KEY"] = "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"
