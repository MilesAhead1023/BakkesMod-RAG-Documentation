# BakkesMod RAG — Debugging & Logging Guide

## Centralized Logging Setup

All logs from the executable are now written to a **single central location** for easy debugging.

### Log Location

When you run `BakkesMod_RAG_GUI.exe`, logs are automatically written to:

```
./logs/bakkesmod_rag.log
```

This path is **relative to where you launch the exe from**.

### Example Paths

| Scenario | Log Path |
|----------|----------|
| Run exe from Desktop | `C:\Users\YourName\Desktop\logs\bakkesmod_rag.log` |
| Run exe from `dist/BakkesMod_RAG_GUI/` | `dist/BakkesMod_RAG_GUI/logs/bakkesmod_rag.log` |
| Run from command line in project root | `./logs/bakkesmod_rag.log` |

---

## What Gets Logged

The centralized log captures **everything**:

### Startup Sequence
```
2026-02-27 12:30:45 | INFO     | bakkesmod_rag | ================================================================================
2026-02-27 12:30:45 | INFO     | bakkesmod_rag | BakkesMod RAG GUI started at 2026-02-27 12:30:45.123456
2026-02-27 12:30:45 | INFO     | bakkesmod_rag | Log file: /full/path/to/logs/bakkesmod_rag.log
2026-02-27 12:30:45 | INFO     | bakkesmod_rag | ================================================================================
```

### RAG Engine Operations
```
2026-02-27 12:31:12 | INFO     | bakkesmod_rag.engine | RAG Query | query="What is a boost?"
2026-02-27 12:31:13 | INFO     | bakkesmod_rag.retrieval | Retrieved 5 documents
2026-02-27 12:31:15 | INFO     | bakkesmod_rag.llm_provider | LLM call: Claude Sonnet (245 tokens in, 189 tokens out)
```

### GUI Activity
```
2026-02-27 12:31:30 | DEBUG    | bakkesmod_rag.cache | Cache hit (92% similarity)
2026-02-27 12:31:45 | DEBUG    | bakkesmod_rag | [GUI] User interaction: clicked "Generate Code"
```

### Errors & Exceptions
```
2026-02-27 12:32:01 | ERROR    | bakkesmod_rag.llm_provider | Failed to connect to Anthropic API
2026-02-27 12:32:01 | CRITICAL | bakkesmod_rag | Uncaught exception: ConnectionError...
```

---

## How to Debug

### 1. **Monitor Logs in Real-Time** (Linux/Mac)

```bash
tail -f logs/bakkesmod_rag.log
```

**Windows (PowerShell):**
```powershell
Get-Content logs/bakkesmod_rag.log -Wait
```

### 2. **Search for Errors**

```bash
grep "ERROR\|CRITICAL" logs/bakkesmod_rag.log
```

### 3. **Filter by Time**

```bash
grep "2026-02-27 12:3[0-5]" logs/bakkesmod_rag.log
```

### 4. **Count Queries**

```bash
grep "RAG Query" logs/bakkesmod_rag.log | wc -l
```

### 5. **View Last 50 Lines**

```bash
tail -50 logs/bakkesmod_rag.log
```

---

## Log Format

Each line follows this format:

```
TIMESTAMP | LEVEL | LOGGER_NAME | MESSAGE
```

| Field | Example | Meaning |
|-------|---------|---------|
| `TIMESTAMP` | `2026-02-27 12:30:45` | Date & time of the event |
| `LEVEL` | `INFO`, `DEBUG`, `ERROR` | Severity: `DEBUG` < `INFO` < `WARNING` < `ERROR` < `CRITICAL` |
| `LOGGER_NAME` | `bakkesmod_rag.engine` | Which module logged this |
| `MESSAGE` | `RAG Query \| query="..."` | The actual log message |

---

## Log Rotation

By default, logs rotate automatically:

- **Max file size:** 10 MB
- **Backup files:** Last 5 rotations kept
  - `bakkesmod_rag.log` (current)
  - `bakkesmod_rag.log.1` (previous)
  - `bakkesmod_rag.log.2` (older)
  - ... up to `.5`

Old logs are kept in the same `./logs/` directory.

---

## Troubleshooting Common Issues

### Issue: "Exe launches but GUI doesn't appear"

**Check logs for:**
```bash
grep "ERROR\|Traceback" logs/bakkesmod_rag.log
```

Look for:
- Missing dependencies (Gradio, Anthropic SDK)
- Port conflicts (default: 7860)
- Missing environment variables (.env file)

### Issue: "Queries failing or slow"

**Check logs for:**
```bash
grep "LLM call\|Retrieval\|Cache" logs/bakkesmod_rag.log
```

Look for:
- LLM provider failures (API key issues, rate limits)
- Slow retrieval times
- Cache hits/misses

### Issue: "Memory usage growing"

**Check logs for:**
```bash
grep "cost_tracker\|tokens\|memory" logs/bakkesmod_rag.log
```

Look for:
- High token counts
- Unbounded cache growth
- Memory leaks in loops

---

## Console Output vs. Log File

| Where | What | Use For |
|-------|------|---------|
| **Console window** | GUI startup messages, user prompts | Quick status checks |
| **GUI Debug Panel** | Last 500 logs (in-memory buffer) | Real-time monitoring while using app |
| **`logs/bakkesmod_rag.log`** | **ALL** logs (persisted to disk) | **Debugging, auditing, post-mortem analysis** |

The **log file is your primary debugging tool** — it never loses data when the app closes.

---

## Example: Full Debug Workflow

1. **Launch exe and reproduce the issue**
   ```bash
   BakkesMod_RAG_GUI.exe
   ```

2. **Monitor logs in real-time (new terminal)**
   ```bash
   tail -f logs/bakkesmod_rag.log
   ```

3. **Interact with GUI to trigger the problem**
   - Submit a query
   - Generate code
   - Reproduce the error

4. **Search logs for errors when problem occurs**
   ```bash
   grep "ERROR\|CRITICAL\|Traceback" logs/bakkesmod_rag.log
   ```

5. **Review context around the error**
   ```bash
   grep -B5 -A5 "ERROR" logs/bakkesmod_rag.log
   ```

---

## Log File Size

Each query + response generates ~500 bytes of logs. At default settings:

- **1 hour of use:** ~5-10 MB
- **1 day of use:** ~100-200 MB (rotating)
- **Old backups:** Kept up to 5 × 10 MB = 50 MB total

After 50 MB total, oldest backups are deleted.

---

## Next Steps

If you encounter an issue:

1. ✓ Stop the exe
2. ✓ Open `logs/bakkesmod_rag.log`
3. ✓ Search for `ERROR` or `CRITICAL`
4. ✓ Share the relevant error block in your bug report

---

**Happy debugging! 🚀**
