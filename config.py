"""
Configuration — single source of truth for model, dataset path, and agent settings.
"""
import os
from dotenv import load_dotenv
load_dotenv()

# ── Nebius Token Factory ──────────────────────────────────────────────
NEBIUS_API_BASE = "https://api.tokenfactory.nebius.com/v1/"
NEBIUS_API_KEY = os.environ.get("NEBIUS_API_KEY", "v1.CmMKHHN0YXRpY2tleS1lMDBmcnc5NG10dGdjZTF2M2ESIXNlcnZpY2VhY2NvdW50LWUwMHdrczM4eXN5YWRlNTF0eTILCOvRhdAGEJyMtR46DAjq1J2bBxDAo-PcAUACWgNlMDA.AAAAAAAAAAHP031Tgw91rBXZsCHsndA-Ln0ft6CzMzfjSijWDfVOMG_3Y4c_10nuCRr2VMOcqJuXHRcbSlutc35-s6HknQgM")

# Qwen3-30B-A3B-Instruct-2507: A Mixture-of-Experts model (30B total, 3B active params).
# Chosen for its strong balance of capability and cost-efficiency:
# - Optimized for chat, reasoning, and tool use - critical for a ReAct agent
# - Fast inference (70 Tok/s) keeps the multi-step agent loop responsive
# - Low cost ($0.10/1M input, $0.30/1M output) allows iterative development
# - Sufficient for both 3-way query routing and multi-step tool-calling
AGENT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
ROUTER_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"

# ── Agent loop ────────────────────────────────────────────────────────
MAX_ITERATIONS = 12  # safety net: 10-15 recommended by the assignment

# ── Dataset ───────────────────────────────────────────────────────────
DATASET_PATH = "data/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"

# ── Persistence (Task 2) ──────────────────────────────────────────────
SQLITE_CHECKPOINT_PATH = "checkpoints.db"
PROFILES_DIR = "profiles"
