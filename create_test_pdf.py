import fitz
import os
from pathlib import Path

# Ensure directory
Path("./data/uploads").mkdir(parents=True, exist_ok=True)

doc = fitz.open()

# Page 1: TEXT only
page1 = doc.new_page()
# Add multiple paragraphs to test chunking.
text = (
    "1. Introduction to the RAG Pipeline\n\n"
    "This is the first paragraph. " * 10 + "\n\n"
    "This is the second paragraph. " * 15 + "\n\n"
    "Here is a short sentence. It has punctuation!"
)
page1.insert_text((50, 50), text, fontsize=12)

# Page 2: MIXED (Text + Image)
page2 = doc.new_page()
page2.insert_text((50, 50), "This page has an image below this text.", fontsize=12)
# Draw a simple rectangle acting as an 'image' for testing, 
# wait, get_images() only finds actual image objects. 
# Let's insert a dummy image. First, create a small raw image.
img_rect = fitz.Rect(50, 100, 150, 200)
# Create a blank image png bytes using fitz
pm = fitz.Pixmap(fitz.csRGB, (0,0,100,100), False)
page2.insert_image(img_rect, pixmap=pm)

# Page 3: IMAGE only (Just a large image taking the whole page)
page3 = doc.new_page()
pm2 = fitz.Pixmap(fitz.csRGB, (0,0,400,600), False)
page3.insert_image(fitz.Rect(0, 0, 400, 600), pixmap=pm2)

out_path = "./data/uploads/test_doc.pdf"
doc.save(out_path)
print(f"Test PDF created at {out_path}")
