import os
import json
import sys
import google.generativeai as genai
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "college-rag")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GENERATION_MODEL = "gemini-2.0-flash"
LOCAL_FACULTY_DATA = r"c:\Users\rohan\OneDrive\Desktop\Work\PROJECTS\COLLEGE RAG PROJECT PRIMARY\eee_faculty_data.json"

SYSTEM_PROMPT = """
You are 'Digital Senior', a helpful senior student at NIT Warangal.

Rules:
1. **CRITICAL:** Answer directly and confidently. Treat the provided Context as your own absolute internal knowledge.
2. **NEVER** reference "the context", "the provided text", "the documents", or "the information provided". Phrases like "Based on the context" or "The text doesn't say" are STRICTLY FORBIDDEN.
3. If the answer is not in your knowledge base (the Context), simply say "I don't have enough information to answer that" or "I'm not sure about that specifically". Do NOT excuse yourself by blaming the missing context.
4. Be concise, professional, and friendly.
5. Do NOT start with greetings like "Hey there". Start directly with the answer.
6. Do NOT end with "Hope this helps".
7. OUTPUT FORMAT: PLAIN TEXT ONLY. No Markdown (no **, *, #). Use simple numbering (1., 2.) or dashes (-) for lists.
8. **restaurant Reviews:** When answering about restaurants:
    - IGNORE the names of the people writing the reviews (e.g. "syed zafar", "Anil Kumar"). They are irrelevant.
    - Focus ONLY on the **Restaurant Name** and the **Substance** of the review (good food, bad service, specific dishes).
    - If the context contains a list of reviews, summarize the general sentiment or specific dishes mentioned for that restaurant.
    - Expected Format: "- [Restaurant Name]: [Summary of what's good/bad] (Price/Location if available)"
"""

class DigitalSeniorBrain:
    def __init__(self):
        print("Initializing Brain...")
        
        # 1. Setup Gemini
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in .env")
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(
            model_name=GENERATION_MODEL,
            system_instruction=SYSTEM_PROMPT
        )
        
        # 2. Setup Pinecone
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not found in .env")
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(INDEX_NAME)
        
        # 3. Setup Embedder
        print(f"Loading Embedder ({EMBEDDING_MODEL})...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        
        # 4. Memory (Simple list for now)
        self.history = []

        # 5. Load Local Faculty Data
        self.local_data = []
        self.local_embeddings = None
        if os.path.exists(LOCAL_FACULTY_DATA):
            print("Loading local faculty data...")
            with open(LOCAL_FACULTY_DATA, 'r', encoding='utf-8') as f:
                self.local_data = json.load(f)
            
            # Pre-compute embeddings for local data
            print(f"Embedding {len(self.local_data)} local items...")
            texts = [item['content'] for item in self.local_data]
            self.local_embeddings = self.embedder.encode(texts, convert_to_tensor=True)
        else:
            print("Warning: Local faculty data file not found.")
        
    def _get_embedding(self, text):
        return self.embedder.encode(text).tolist()

    def classify_intent(self, query):
        """
        Decides if the query needs RAG or is just chit-chat.
        Returns: { "type": "rag_search" | "chit_chat", "category": "..." }
        """
        history_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.history[-3:]])
        
        # Define the hierarchy of knowledge available in the system
        # This helps the LLM understand what specific topics fall under which category
        knowledge_hierarchy = {
            "Faculty": ["Specific Professors (e.g., D V S S Siva Sarma)", "Research Areas", "Teaching Style"],
            "Academics": ["Attendance Policy", "UG Regulations", "Academic Calendar", "Student Feedback"],
            "Food": ["Canteens", "Messes (e.g. IFC - B)", "Menu"],
            "Hostels": ["Hostel Blocks (e.g., Azad, Bose)", "General Rules", "Facilities", "Wardens", "Repairs (LAN, Electrician, Plumbing)", "Issues"],
            "Placements": ["Company Details", "Placement Statistics"],
            "Campus_Life": ["Fests", "Clubs", "Events"],
            "Admin": ["Fees & Scholarships", "Documents & Transcripts"],
            "Facilities": ["Library", "Health Centre", "Sports"],
            "Guides": ["City Guide", "Freshers Guide"],
            "Internships": ["Company Specific Experiences (e.g. Amazon, Microsoft)", "Process", "Questions"]
        }
        
        hierarchy_str = json.dumps(knowledge_hierarchy, indent=2)
        
        prompt = f"""
        You are the Router for a college chatbot. Your job is to classify the user's intent into one of the available categories or identify it as chit-chat.
        
        User Query: "{query}"
        Recent History:
        {history_context}
        
        Available Knowledge Hierarchy (Category -> Subcategories/Topics):
        {hierarchy_str}
        
        Instructions:
        1. Analyze the User Query to understand what they are looking for.
        2. Match their intent to the most relevant 'Category' from the hierarchy above. Use the subcategories as strong hints.
           - Example: "When is the exam?" -> Matches 'Academic Calendar' -> Category: "Academics"
           - Example: "Who is the warden of Azad Hall?" -> Matches 'Wardens'/'Hostel Blocks' -> Category: "Hostels"
           - Example: "Tell me about Prof Siva Sarma" -> Matches 'Specific Professors' -> Category: "Faculty"
           - Example: "What is for dinner in IFC B?" -> Matches 'Messes' -> Category: "Food", filters: {{ "filter": "IFC - B" }}
           - Example: "Lan repair number?" -> Matches 'Repairs' -> Category: "Hostels"
        3. If the user asks about a specific topic not explicitly listed but related to a category (e.g., "Mess menu" relates to "Food"), choose that category.
        4. Extract 'filter' for:
           - 'Internships' (Target: Company Name)
           - 'Food' (Target: Mess Name)
           - 'Academics' (Target: Specific Course Name).
             - Do NOT extract generic terms like "Credits", "Syllabus", "Exams", "Regulations", "Grades" as filters. Set "filter": null for general queries.
             - Example: "Syllabus for Digital Electronics" -> Category: "Academics", filters: {{ "filter": "Digital Electronics" }}
             - Example: "How are credits assigned?" -> Category: "Academics", filters: {{ "filter": null }}
        5. If the user says "Hi", "Thanks", "Bye", or asks a general question NOT about the college (e.g. "What is 2+2?"), output type="chit_chat".
        
        Output JSON ONLY:
        {{ "type": "rag_search" | "chit_chat", "category": "CategoryName" | null, "filters": {{ "filter": "FILTER_VALUE" | null }} }}
        """
        
        try:
            response = self.model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            return json.loads(response.text)
        except Exception as e:
            print(f"Router Error: {e}")
            return {"type": "chit_chat", "category": None}

    def search_db(self, query, category, filters=None):
        """
        Searches Pinecone for context within a specific category and optional filters.
        """
        # Generate embedding
        vector = self._get_embedding(query)
        
        # Construct metadata filter
        meta_filter = {}
        if category:
            meta_filter["category"] = category

        # SAFETY CHECK: Only allow filters for Internships and Food
        # This overrides any LLM hallucination for categories like Academics
        valid_filter_categories = ["Internships", "Food", "Academics"]
        if category not in valid_filter_categories and filters:
             filters["filter"] = None

        if filters and filters.get("filter"):
            meta_filter["filter"] = filters["filter"].upper()  # Ensure matches ingestion format

        print(f"Searching DB for '{query}' in category '{category}' with filters {meta_filter}...")

        contexts = []

        # 1. Search Pinecone (Remote DB)
        try:
            results = self.index.query(
                vector=vector,
                top_k=5,
                include_metadata=True,
                filter=meta_filter if meta_filter else None
            )
            
            for match in results.matches:
                if match.score > 0.3:
                    text_to_use = match.metadata.get("context_text", match.metadata.get("text", ""))
                    contexts.append(text_to_use)
        except Exception as e:
            print(f"Pinecone Search Error: {e}")

        # 2. Search Local Data (In-Memory)
        # Only if category is Faculty or generic/None (to be safe)
        if self.local_embeddings is not None and (category == "Faculty" or category is None):
            print("Searching local faculty data...")
            query_embedding = self.embedder.encode(query, convert_to_tensor=True)
            hits = util.semantic_search(query_embedding, self.local_embeddings, top_k=3)
            
            # Hits is a list of lists (one per query). We only have one query.
            for hit in hits[0]:
                if hit['score'] > 0.3: # Threshold
                    idx = hit['corpus_id']
                    content = self.local_data[idx]['content']
                    contexts.append(content)
                
        return "\n\n".join(contexts)

    def generate_response(self, query):
        """
        Main function to handle a user query.
        """
        # 1. Classify Intent
        intent = self.classify_intent(query)
        print(f"Intent: {intent}")
        
        context = ""
        if intent["type"] == "rag_search" and intent["category"]:
            filters = intent.get("filters")
            context = self.search_db(query, intent["category"], filters)
            # Safe print for Windows terminals (Direct Byte Write)
            try:
                header = "\n--- RETRIEVED CONTEXT START ---\n"
                footer = "\n--- RETRIEVED CONTEXT END ---\n"
                sys.stdout.buffer.write(header.encode('utf-8'))
                sys.stdout.buffer.write(context.encode('utf-8', 'replace'))
                sys.stdout.buffer.write(footer.encode('utf-8'))
            except Exception:
                 print(f"\n--- RETRIEVED CONTEXT START ---\n(Context print failed)\n--- RETRIEVED CONTEXT END ---\n")
            
        # 2. Generate Answer
        history_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.history[-5:]])
        
        system_prompt = """
        You are 'Digital Senior', a helpful senior student at NIT Warangal.
        
        Rules:
        1. **CRITICAL:** Answer directly and confidently. Treat the provided Context as your own absolute internal knowledge.
        2. **NEVER** reference "the context", "the provided text", "the documents", or "the information provided". Phrases like "Based on the context" or "The text doesn't say" are STRICTLY FORBIDDEN.
        3. If the answer is not in your knowledge base (the Context), simply say "I don't have enough information to answer that" or "I'm not sure about that specifically". Do NOT excuse yourself by blaming the missing context.
        4. Be concise, professional, and friendly.
        5. Do NOT start with greetings like "Hey there". Start directly with the answer.
        6. Do NOT end with "Hope this helps".
        7. OUTPUT FORMAT: PLAIN TEXT ONLY. No Markdown (no **, *, #). Use simple numbering (1., 2.) or dashes (-) for lists.
        8. **Restaurant Reviews:** When answering about restaurants:
            - IGNORE the names of the people writing the reviews (e.g. "syed zafar", "Anil Kumar"). They are irrelevant.
            - Focus ONLY on the **Restaurant Name** and the **Substance** of the review (good food, bad service, specific dishes).
            - If the context contains a list of reviews, summarize the general sentiment or specific dishes mentioned for that restaurant.
            - Expected Format: "- [Restaurant Name]: [Summary of what's good/bad] (Price/Location if available)"
        """
        
        user_prompt = f"""
        Context:
        {context}
        
        Chat History:
        {history_context}
        
        User: {query}
        """
        
        response = self.model.generate_content(
            contents=[
                {"role": "user", "parts": [system_prompt + "\n" + user_prompt]}
            ]
        )
        
        answer = response.text.strip()
        
        # 3. Update Memory
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "model", "content": answer})
        
        return answer

if __name__ == "__main__":
    # Simple CLI Test
    brain = DigitalSeniorBrain()
    print("\n--- Digital Senior CLI (Type 'quit' to exit) ---")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["quit", "exit"]:
            break
        
        response = brain.generate_response(user_input)
        
        # Safe print for response (Direct Byte Write)
        prefix = "Senior: "
        sys.stdout.buffer.write(prefix.encode('utf-8'))
        sys.stdout.buffer.write(response.encode('utf-8', 'replace'))
        sys.stdout.buffer.write(b"\n")
