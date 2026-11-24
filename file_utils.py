from pathlib import Path
import fitz
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def read_pdf(path):
    doc=fitz.open(path)
    text="\n".join(page.get_text() for page in doc)
    doc.close()
    return text

def write_pdf(text,path):
    styles=getSampleStyleSheet()
    doc=SimpleDocTemplate(str(path),pagesize=A4,rightMargin=40,leftMargin=40,topMargin=40,bottomMargin=40)
    story=[]
    for line in text.splitlines():
        safe=line.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        story.append(Paragraph(safe,styles["BodyText"]))
        story.append(Spacer(1,5))
    doc.build(story)

def read_excel(path):
    return pd.read_excel(path, sheet_name=None)

def write_excel(sheets,path):
    with pd.ExcelWriter(path,engine="xlsxwriter") as writer:
        for name,df in sheets.items():
            df.to_excel(writer,sheet_name=str(name)[:31],index=False)
