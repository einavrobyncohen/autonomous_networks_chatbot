import os
from pypdf import PdfReader
import re
from typing import List, Dict

def clean_text(text: str) -> str:
    """Enhanced text cleaning with better preservation of technical terms."""
    # Remove extra whitespace and newlines while preserving sentence boundaries
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep technical formatting
    text = re.sub(r'[^\w\s.,;:()\-/]', '', text)
    # Normalize whitespace around punctuation
    text = re.sub(r'\s*([.,;:])\s*', r'\1 ', text)
    # Remove multiple periods
    text = re.sub(r'\.{2,}', '.', text)
    return text.strip()

def extract_sections(text: str) -> List[Dict[str, str]]:
    """Split text into semantic sections based on common document patterns."""
    sections = []
    current_section = ""
    current_title = "Introduction"  # Default section title
    
    # Common section header patterns in technical documents
    section_patterns = [
        r'^(?:\d+\.)?\s*(?:Introduction|Overview|Background|Implementation|Architecture|Components|Conclusion)',
        r'^(?:\d+\.)?\s*[A-Z][A-Za-z\s]{2,50}$'
    ]
    
    combined_pattern = '|'.join(f'({pattern})' for pattern in section_patterns)
    
    lines = text.split('\n')
    
    for line in lines:
        match = re.match(combined_pattern, line.strip())
        if match:
            # Save previous section if it exists
            if current_section.strip():
                sections.append({
                    'title': current_title,
                    'content': clean_text(current_section)
                })
            current_title = line.strip()
            current_section = ""
        else:
            current_section += line + " "
    
    # Add the last section
    if current_section.strip():
        sections.append({
            'title': current_title,
            'content': clean_text(current_section)
        })
    
    return sections

def convert_pdfs_to_text():
    """Convert PDFs to structured text files with better organization."""
    if not os.path.exists('processed_data'):
        os.makedirs('processed_data')
    
    for filename in os.listdir('data'):
        if filename.endswith('.pdf'):
            try:
                print(f'Processing {filename}...')
                reader = PdfReader(f'data/{filename}')
                
                # Extract and combine text from all pages
                full_text = ''
                for page in reader.pages:
                    full_text += page.extract_text() + '\n'
                
                # Extract sections
                sections = extract_sections(full_text)
                
                # Save structured content
                output_filename = f'processed_data/{filename[:-4]}.txt'
                with open(output_filename, 'w', encoding='utf-8') as f:
                    for section in sections:
                        f.write(f"# {section['title']}\n\n")
                        f.write(f"{section['content']}\n\n")
                
                print(f'Successfully converted {filename}')
                
            except Exception as e:
                print(f'Error processing {filename}: {str(e)}')

if __name__ == "__main__":
    convert_pdfs_to_text()