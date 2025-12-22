import os
import pypdf

def read_text_file(filepath: str) -> str:
    """Reads content from a text file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def read_pdf_file(filepath: str) -> str:
    """Extracts text from a PDF file."""
    text = ""
    try:
        reader = pypdf.PdfReader(filepath)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

def chunk_text(text: str, max_chars: int = 2000) -> list[str]:
    """
    Splits text into manageable chunks for the LLM, trying to respect sentence/paragraph boundaries.
    """
    chunks = []
    current_chunk = []
    current_length = 0
    
    paragraphs = text.split('\n\n')
    
    for para in paragraphs:
        if current_length + len(para) > max_chars:
            # Join current chunk and reset
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_length = len(para)
        else:
            current_chunk.append(para)
            current_length += len(para)
            
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
        
    return chunks

if __name__ == "__main__":
    text = "Para 1\n\nPara 2\n\nPara 3"
    print(chunk_text(text, 10))
