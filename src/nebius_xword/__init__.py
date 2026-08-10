"""Nebius-XWord: an LLM crossword-solving agent on Nebius AI Studio.

The core grid/solver modules are dependency-free; import
``nebius_xword.agent`` explicitly when you need the LLM agent (it pulls in
``openai`` and ``python-dotenv``).
"""

from .grid import BLOCK, EMPTY, Grid, Puzzle, Slot, load_puzzle

__version__ = "0.1.0"
__all__ = ["BLOCK", "EMPTY", "Grid", "Puzzle", "Slot", "load_puzzle", "__version__"]
