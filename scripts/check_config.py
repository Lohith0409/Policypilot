import sys
sys.path.append(".")  # lets Python find the src package from project root

from src.config import GROQ_API_KEY, QDRANT_URL

print("Config loaded successfully.")
print(f"Groq key starts with: {GROQ_API_KEY[:8]}...")
print(f"Qdrant URL: {QDRANT_URL}")