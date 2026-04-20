import easyocr
import numpy as np

print("Initializing EasyOCR with GPU...")
try:
    reader = easyocr.Reader(['en'], gpu=True)
    print("EasyOCR initialized successfully!")
    
    # Create a random noise image (100x100)
    print("Testing readtext on dummy image...")
    dummy_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    results = reader.readtext(dummy_img)
    print("Readtext completed! Results:", results)
except Exception as e:
    print(f"FAILED: {e}")
