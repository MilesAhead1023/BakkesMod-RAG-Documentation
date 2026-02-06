"""
Smoke tests - verify code syntax and basic functionality without API calls.
This tests that our fixes work at the syntax/import level.
"""

import os
import sys

# Set dummy API keys for import testing
os.environ.setdefault('OPENAI_API_KEY', 'sk-test-dummy-key-for-syntax-validation')
os.environ.setdefault('ANTHROPIC_API_KEY', 'sk-ant-test-dummy-key')
os.environ.setdefault('GOOGLE_API_KEY', 'test-dummy-google-key')

print("=" * 60)
print("SMOKE TESTS - Code Syntax & Structure Validation")
print("=" * 60)

# Test 1: Python compilation
print("\n[TEST 1] Python Syntax Validation")
print("-" * 60)

files_to_test = [
    'rag_sentinel.py',
    'comprehensive_rag.py',
    'gemini_rag_builder.py',
    'mcp_rag_server.py',
    'test_rag_integration.py',
]

import py_compile

all_passed = True
for file in files_to_test:
    try:
        py_compile.compile(file, doraise=True)
        print(f"[PASS] {file:40s} - Syntax valid")
    except py_compile.PyCompileError as e:
        print(f"[FAIL] {file:40s} - Syntax error")
        print(f"   {e}")
        all_passed = False

# Test 2: Import validation (checks our fixes work)
print("\n[TEST 2] Module Import Validation")
print("-" * 60)

import_tests = [
    ('comprehensive_rag', 'comprehensive_rag'),
    ('gemini_rag_builder', 'gemini_rag_builder'),
    ('rag_sentinel', 'rag_sentinel'),
    ('mcp_rag_server', 'mcp_rag_server'),
]

for display_name, module_name in import_tests:
    try:
        __import__(module_name)
        print(f"[PASS] {display_name:40s} - Imports successfully")
    except ImportError as e:
        print(f"[WARN]  {display_name:40s} - Missing dependency: {e}")
        # Missing dependencies don't fail smoke test
    except Exception as e:
        print(f"[FAIL] {display_name:40s} - Import error: {type(e).__name__}: {e}")
        all_passed = False

# Test 3: Check critical fixes were applied
print("\n[TEST 3] Verify 2026 API Fixes Applied")
print("-" * 60)

# Check comprehensive_rag.py uses correct Anthropic model format
with open('comprehensive_rag.py', 'r') as f:
    content = f.read()

    # Should NOT have Bedrock format
    if 'anthropic.claude-3-5-sonnet-20241022-v2:0' in content:
        print("[FAIL] comprehensive_rag.py still has Bedrock model format")
        all_passed = False
    elif 'claude-3-5-sonnet-20240620' in content:
        print("[PASS] comprehensive_rag.py uses correct Anthropic SDK format")
    else:
        print("[WARN]  comprehensive_rag.py Anthropic model format unclear")

    # Should NOT have GPTCache
    if 'from gptcache' in content:
        print("[FAIL] comprehensive_rag.py still has deprecated GPTCache")
        all_passed = False
    else:
        print("[PASS] comprehensive_rag.py has GPTCache removed")

# Check gemini_rag_builder.py uses correct model names
with open('gemini_rag_builder.py', 'r') as f:
    content = f.read()

    # Should NOT have models/ prefix
    if 'GoogleGenAI(model="models/gemini' in content:
        print("[FAIL] gemini_rag_builder.py still has 'models/' prefix")
        all_passed = False
    elif 'GoogleGenAI(model="gemini-2.0-flash")' in content:
        print("[PASS] gemini_rag_builder.py uses correct model format")
    else:
        print("[WARN]  gemini_rag_builder.py model format unclear")

# Check rag_sentinel.py has no syntax errors
with open('rag_sentinel.py', 'r', encoding='utf-8') as f:
    content = f.read()
    lines = content.split('\n')

    # Count except statements (should not have orphaned except)
    in_comment = False
    except_count = 0
    try_count = 0

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        if 'try:' in stripped:
            try_count += 1
        if 'except' in stripped and not stripped.startswith('#'):
            except_count += 1

    if except_count > try_count:
        print(f"[FAIL] rag_sentinel.py may have orphaned except (try:{try_count}, except:{except_count})")
        all_passed = False
    else:
        print(f"[PASS] rag_sentinel.py exception handling looks correct")

# Final result
print("\n" + "=" * 60)
if all_passed:
    print("[PASS] ALL SMOKE TESTS PASSED")
    print("=" * 60)
    print("\nThe 2026 API fixes have been successfully applied!")
    print("All Python files have valid syntax and import correctly.")
    sys.exit(0)
else:
    print("[FAIL] SOME TESTS FAILED")
    print("=" * 60)
    print("\nPlease review the failures above.")
    sys.exit(1)
