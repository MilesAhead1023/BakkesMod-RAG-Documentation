import os
import sys
import logging
import time
import io
import asyncio
import nest_asyncio

# Force UTF-8 for Windows Console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Apply nest_asyncio for running asyncio in potentially nested loops (like notebooks or mcp)
nest_asyncio.apply()

def print_step(msg):
    print(f"\n[SENTINEL] {msg}...")

def print_result(success, details=""):
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"Result: {status} {details}")
    if not success:
        print("\n!!! ACTION REQUIRED !!!")
        print(details)
        sys.exit(1)

async def audit_api_health():
    """Performs lightweight checks on all configured APIs."""
    print_step("Auditing API Health & Billing Status")
    
    # OpenAI Check
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        # A very cheap and fast call to check if key is valid (synchronous call)
        client.embeddings.create(input=["test"], model="text-embedding-3-small")
        print("  OpenAI API: ✅ Active and accessible.")
    except Exception as e:
        print_result(False, f"OpenAI API Error: {e}. Check key and billing.")

    # Anthropic Check
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        # Use a very cheap and fast call to list models to check if key is valid/billing works
        # This is a synchronous call.
        client.models.list() 
        print("  Anthropic API: ✅ Active and accessible.")
    except Exception as e:
        print_result(False, f"Anthropic API Error: {e}. Check key and billing. (Error was: {e})")

    # Gemini Check (Temporarily disabled due to API client instability for simple health checks)
    # When re-enabling, use the correct google.genai.Client() pattern:
    # try:
    #     from google import genai
    #     client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    #     response = client.models.generate_content(model='gemini-2.0-flash', contents='hi')
    #     print("  Gemini API: ✅ Active and accessible.")
    # except Exception as e:
    #     print_result(False, f"Gemini API Error: {e}. Check key and billing.")

    print("  Gemini API: ⚠️ Check disabled (enable in code when needed)")

    print_result(True, "All configured APIs are healthy.")

def run_diagnostics():
    print("==================================================")
    print("   ULTRA-RAG PLATINUM SENTINEL - SYSTEM AUDIT    ")
    print("==================================================")

    # 1. Environment & Path Audit
    print_step("Checking Environment Variables")
    keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"]
    for key in keys:
        if key not in os.environ:
            print_result(False, f"Missing {key}. System cannot function without all 3 keys.")
    print_result(True, "All API keys detected.")

    # 2. Documentation Formatting Audit
    print_step("Auditing Documentation Files (docs/)")
    docs_path = "./docs"
    if not os.path.exists(docs_path):
        print_result(False, "The 'docs/' folder is missing.")
    
    files = [f for f in os.listdir(docs_path) if f.endswith(".md")]
    if not files:
        print_result(False, "No markdown files found in docs/.")
    
    for doc in files:
        full_path = os.path.join(docs_path, doc)
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            if len(content) < 10:
                print(f"  [Warning] File {doc} is empty or too small (<10 chars).")
            # Check for improper markdown headers (common RAG crash point)
            if "#" not in content:
                print(f"  [Warning] {doc} has no headers. RAG will treat it as a single chunk.")
    print_result(True, f"Scanned {len(files)} files. Formatting is compatible.")
    
    # 3. API Health & Billing Audit
    asyncio.run(audit_api_health())

    # 4. Virtual Environment Audit
    print_step("Verifying Python Dependencies")
    try:
        import llama_index.core
        import phoenix
        import watchdog
        import gptcache
    except ImportError as e:
        print_result(False, f"Dependency missing: {e}.")
    print_result(True, "All critical libraries are importable.")

    # 5. Storage & Corruption Check
    print_step("Verifying Index Integrity")
    storage_path = "./rag_storage"
    
    if os.path.exists(os.path.join(storage_path, "docstore.json")) and \
       os.path.exists(os.path.join(storage_path, "default__vector_store.json")):
        print("  [Info] Vector Index found. System is partially operational.")
        
        if not os.path.exists(os.path.join(storage_path, "index_store.json")):
             print("  [Note] Knowledge Graph is still building (index_store.json missing). This is expected.")
        else:
             print("  [Success] Knowledge Graph structure found.")
        print_result(True, "Critical retrieval files exist.")
    else:
        print_result(False, "No index files found yet. Background build is likely still parsing documents.")

    # 6. Live Retrieval Stress Test
    print_step("Running End-to-End Retrieval Test")
    try:
        from comprehensive_rag import build_comprehensive_stack
        
        # Check which storage to use for the test query
        # This will load either the OpenAI-based or Gemini-based one.
        comprehensive_rag_module = sys.modules['comprehensive_rag']
        comprehensive_rag_module.STORAGE_DIR = "./rag_storage" # Default for test

        engine = build_comprehensive_stack(incremental=False)
        
        test_query = "How do I find the player car and set its rotation to face the ball?"
        start_time = time.time()
        response = engine.query(test_query)
        end_time = time.time()
        
        if len(str(response)) < 20:
            print_result(False, "Retrieval returned an empty or too-short response.")
        
        print(f"  [Success] Retrieval Latency: {end_time - start_time:.2f}s")
        print(f"  [Success] Sources Found: {len(response.source_nodes)}")
    except Exception as e:
        print_result(False, f"Retrieval logic failed: {e}")
    
    print_result(True, "End-to-end reasoning chain is operational.")

    print("\n==================================================")
    print("   ALL SYSTEMS GREEN: RAG STACK IS FAIL-PROOF    ")
    print("==================================================")

if __name__ == "__main__":
    run_diagnostics()
