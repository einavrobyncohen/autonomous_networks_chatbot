
import os
from pypdf import PdfReader
import re

def clean_text(text):
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep periods and commas
    text = re.sub(r'[^\w\s.,]', '', text)
    return text.strip()

def convert_pdfs_to_text():
    # Create processed_data directory if it doesn't exist
    if not os.path.exists('processed_data'):
        os.makedirs('processed_data')
    
    # Process each PDF in the data directory
    for filename in os.listdir('data'):
        if filename.endswith('.pdf'):
            try:
                # Read PDF
                reader = PdfReader(f'data/{filename}')
                
                # Extract text from each page
                text = ''
                for page in reader.pages:
                    text += page.extract_text() + '\n'
                
                # Clean the text
                text = clean_text(text)
                
                # Save to text file
                output_filename = f'processed_data/{filename[:-4]}.txt'
                with open(output_filename, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                print(f'Successfully converted {filename}')
                
            except Exception as e:
                print(f'Error processing {filename}: {str(e)}')

if __name__ == "__main__":
    convert_pdfs_to_text()