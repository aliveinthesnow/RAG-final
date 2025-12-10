import json
import re

INPUT_FILE = r"c:\Users\rohan\OneDrive\Desktop\Work\PROJECTS\COLLEGE RAG PROJECT PRIMARY\eee_faculty_data.txt"
OUTPUT_FILE = r"c:\Users\rohan\OneDrive\Desktop\Work\PROJECTS\COLLEGE RAG PROJECT PRIMARY\eee_faculty_data.json"

def parse_faculty_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by "Prof." but keep the delimiter. 
    # The first split might be empty if the file starts with "Prof."
    raw_profiles = re.split(r'(?=Prof\.)', content)
    
    faculty_list = []
    
    for profile_text in raw_profiles:
        profile_text = profile_text.strip()
        if not profile_text:
            continue
            
        lines = profile_text.split('\n')
        if not lines:
            continue

        # Extract Name
        name_line = lines[0].strip()
        # Remove "Prof." prefix for the name field, but keep it for content
        name = name_line.replace("Prof.", "").strip()
        
        # Generate ID
        name_slug = name.lower().replace('.', '').replace(' ', '_')
        fac_id = f"fac_{name_slug}"
        
        # Construct Content
        # We want the content to be the full text block as it appears in the file
        # to preserve context for the RAG.
        # We can lightly format it or just keep it as is.
        # The user requested "One professor = One chunk".
        
        # Metadata
        metadata = {
            "category": "Faculty",
            "sub_category": name,
            "filter": ""
        }
        
        faculty_obj = {
            "id": fac_id,
            "content": profile_text,
            "metadata": metadata
        }
        
        faculty_list.append(faculty_obj)
        
    return faculty_list

def main():
    print(f"Reading from {INPUT_FILE}...")
    data = parse_faculty_data(INPUT_FILE)
    
    print(f"Parsed {len(data)} faculty profiles.")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
        
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
