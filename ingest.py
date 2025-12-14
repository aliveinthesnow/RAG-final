import json
import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

# --- CONFIGURATION ---
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "college-rag")
DATA_FILE = "MASTER_DATA.json"
MODEL_NAME = "all-MiniLM-L6-v2"

def load_data():
    """Loads the master JSON data."""
    print(f"Loading data from {DATA_FILE}...")
    combined_data = []
    
    # 1. Load Master Data
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} items from MASTER_DATA.")
        combined_data.extend(data)
    except FileNotFoundError:
        print(f"Error: {DATA_FILE} not found.")
    except json.JSONDecodeError:
        print(f"Error: {DATA_FILE} is not valid JSON.")

    return combined_data

def init_pinecone():
    """Initializes Pinecone client and index."""
    if not PINECONE_API_KEY:
        print("Error: PINECONE_API_KEY not found in .env")
        return None, None
    
    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists
    existing_indexes = [index.name for index in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        print(f"Index '{INDEX_NAME}' not found. Please create it in the Pinecone console.")
        print(f"   Dimension: 384, Metric: Cosine")
        return None, None
        
    index = pc.Index(INDEX_NAME)
    print(f"Connected to index: {INDEX_NAME}")
    return pc, index

def generate_embeddings(data, model):
    """Generates embeddings for the content with auto-chunking."""
    print(f"Loading model: {MODEL_NAME}...")
    embedder = SentenceTransformer(MODEL_NAME)
    
    # Standard splitter for normal text
    standard_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    print("Generating embeddings...")
    items_to_upload = []
    
    for item in data:
        # Skip empty items
        if not item.get("content"):
            continue
            
        chunks = []
        
        # SPECIAL HANDLING: Structure-Aware Splitting for ALL items
        # We apply the Universal Hybrid Strategy globally because it has a safe fallback.
        print(f"   -> Applying Universal Hybrid Strategy for {item['id']}...")
            
        # 1. Universal Structural Split
        content = "\n" + item["content"]
        
        # SPECIAL HANDLING: Parent-Child / Section-Based Retrieval
        PARENT_CHILD_IDS = [
            "electrical_syllabus", 
            "ug_regulations_24_25", 
            "academic_calendar_details",
            "placement_stats_2024_25",
            "mess_menu_ifc_b",
            "hostels_primary_data"
        ]
        
        # Apply Parent-Child to "Hostels" category automatically.
        is_parent_child = (
            item['id'] in PARENT_CHILD_IDS or 
            item.get('metadata', {}).get('category') == 'Internships' or 
            item.get('metadata', {}).get('category') == 'Hostels' or
            item.get('metadata', {}).get('subcategory') == 'Syllabus'
        )
        
        if is_parent_child:
            print(f"   -> Applying Section-Based Parent Retrieval for '{item['id']}'...")
            
            # 1. Split by Headers to get full Sections (The "Parents")
            # REGEX UPDATE: More specific header pattern to avoid infinite loops on large files
            header_pattern = r'(\n##\s+|\n\d+\.\s+|\n\d+\)\s+)'
            raw_parts = re.split(header_pattern, content)
            
            # Reconstruct sections (header + content)
            course_sections = []
            current_section = ""
            
            for part in raw_parts:
                # Check if this part is a header delimiter
                if re.match(header_pattern, part) or re.match(r'^(##\s+|\d+\.\s+|\d+\)\s+)', part):
                    if current_section.strip():
                        course_sections.append(current_section.strip())
                    current_section = part # Start new section
                else:
                    current_section += part # Append content to header
            
            if current_section.strip():
                course_sections.append(current_section.strip())
                
            print(f"      -> Found {len(course_sections)} logical parent sections.")

            # 2. Process each Section
            for section_text in course_sections:
                # This 'section_text' is the PARENT context.
                section_chunks = standard_splitter.split_text(section_text)
                
                for i, chunk_text in enumerate(section_chunks):
                    vector_id = f"{item['id']}_{hash(chunk_text)}"
                    
                    metadata = {
                        "text": chunk_text, # What is searched/matched
                        "context_text": section_text, # The FULL "Parent" context to show the user
                        "source_id": item['id'],
                        "category": item['metadata']['category'],
                        "subcategory": item['metadata'].get('subcategory', ''),
                        "filter": item['metadata'].get('filter', ''),
                        "chunk_index": i
                    }
                    
                    # Generate embedding
                    embedding = embedder.encode(chunk_text).tolist()
                    
                    items_to_upload.append({
                        "id": vector_id,
                        "values": embedding,
                        "metadata": metadata
                    })
                    
        else:
            # STANDARD LOGIC FOR ALL OTHER ITEMS (Legacy/Simple Chunking)
            regex_pattern = r'(\n##\s+|\n\d+\.\s+|\n\d+\)\s+)'
            parts = re.split(regex_pattern, content)
            
            structural_chunks = []
            current_chunk = ""
            
            for part in parts:
                if re.match(r'(\n##\s+|\n\d+\.\s+|\n\d+\)\s+)', part):
                    if current_chunk.strip():
                        structural_chunks.append(current_chunk.strip())
                    current_chunk = part.strip()
                else:
                    current_chunk += part
            
            if current_chunk.strip():
                structural_chunks.append(current_chunk.strip())
                
            print(f"      -> Found {len(structural_chunks)} structural sections.")
            
            # 2. Recursive Refinement
            for s_chunk in structural_chunks:
                if len(s_chunk) < 1000:
                    chunks.append(s_chunk)
                else:
                    sub_chunks = standard_splitter.split_text(s_chunk)
                    chunks.extend(sub_chunks)
            
            for i, chunk in enumerate(chunks):
                # CONTEXT FIX: For Food/facilities, prepend the name so it's never lost in sub-chunks (like Reviews)
                if item['metadata'].get('category') == 'Food' or item['metadata'].get('category') == 'Facilities':
                    name_context = item['metadata'].get('sub_category') or item['metadata'].get('filter') or ""
                    if name_context and name_context not in chunk:
                        chunk = f"**{name_context}**\n{chunk}"

                embedding = embedder.encode(chunk).tolist()
                
                base_id = item.get("id")
                if not base_id:
                    print(f"Warning: Item missing ID. Skipping: {item['metadata'].get('sub_category')}")
                    continue
                    
                chunk_id = f"{base_id}_chunk_{i}"
                
                vector = {
                    "id": chunk_id,
                    "values": embedding,
                    "metadata": {
                        "category": item["metadata"]["category"],
                        "subcategory": item["metadata"].get("subcategory", ""), # SAFELY GET
                        "filter": item["metadata"].get("filter", ""), # ADDED FILTER
                        "text": chunk, 
                        "context_text": chunk, 
                        "source_id": base_id, 
                        "chunk_index": i
                    }
                }
                items_to_upload.append(vector)
        
    print(f"Generated {len(items_to_upload)} vectors from {len(data)} items.")
    return items_to_upload

def upsert_data(index, vectors):
    """Uploads vectors to Pinecone in batches."""
    BATCH_SIZE = 100
    total_vectors = len(vectors)
    print(f"Uploading {total_vectors} vectors to Pinecone...")
    
    for i in range(0, total_vectors, BATCH_SIZE):
        batch = vectors[i:i + BATCH_SIZE]
        index.upsert(vectors=batch)
        print(f"   Uploaded batch {i // BATCH_SIZE + 1}/{(total_vectors + BATCH_SIZE - 1) // BATCH_SIZE}")
        
    print("Ingestion complete!")

def run_ingestion():
    # 1. Load Data
    data = load_data()
    if not data: return

    # 2. Init Pinecone
    pc, index = init_pinecone()
    if not index: return

    # 3. Generate Embeddings
    vectors = generate_embeddings(data, MODEL_NAME)
    
    # 4. Upload
    if vectors:
        upsert_data(index, vectors)
    else:
        print("No valid vectors to upload.")

if __name__ == "__main__":
    run_ingestion()
