import numpy as np
import json
import os
from typing import Dict


class GrammarState:
    """Tracks the current state in the grammar graph."""

    def __init__(self, state_name: str):
        self.name = state_name


class GrammarPruner:
    """
    Prunes logits based on a grammar graph (FSA).
    Ensures generated tokens follow valid transitions.
    """

    def __init__(
        self, grammar_name: str = "json", grammar_dir: str = "core/grammar/grammars"
    ):
        self.grammar = self._load_grammar(grammar_name, grammar_dir)
        self.start_state = self.grammar.get("start_state", "START")

        # Token mapping (Mocked for now since we don't have full tokenizer access)
        # In real integration, we'd inject the tokenizer to map "{" -> TokenID
        self.token_map = {}

    def _load_grammar(self, name: str, directory: str) -> Dict:
        path = os.path.join(directory, f"{name}.json")
        if not os.path.exists(path):
            # Fallback for tests if running from root
            path = os.path.join(os.getcwd(), directory, f"{name}.json")

        with open(path, "r") as f:
            return json.load(f)

    def get_initial_state(self) -> GrammarState:
        return GrammarState(self.start_state)

    def prune_logits(
        self, logits: np.ndarray, state: GrammarState, token_to_id_fn
    ) -> np.ndarray:
        """
        Mask logits that are invalid given the current state.

        Args:
           logits: [VocabSize]
           state: Current GrammarState
           token_to_id_fn: Function mapping string token -> ID (or list of IDs)

        Returns:
           masked_logits
        """
        # 1. Get allowed transitions
        transitions = self.grammar["states"].get(state.name, [])

        if not transitions:
            return logits  # Terminal or unknown state

        # 2. Create mask (initialize to -inf)
        mask = np.full_like(logits, -float("inf"))
        allowed_any = False

        for trans in transitions:
            if "token" in trans:
                target_token = trans["token"]
                # Convert to ID
                t_id = token_to_id_fn(target_token)
                if t_id is not None:
                    if isinstance(t_id, list):
                        mask[t_id] = logits[t_id]
                    else:
                        mask[t_id] = logits[t_id]

            elif "rule" in trans:
                # Handle special rules like ANY_STRING or NUMBER
                # For this prototype, we'll assume "rule" implies we allow
                # a broad set of tokens.
                # If ANY_STRING is active, we basically unmask everything
                # (except maybe structural tokens if we want to be strict,
                # but simplistic approach is allow all).
                allowed_any = True

        if allowed_any:
            return logits  # Allow everything

        return mask

    def update_state(self, state: GrammarState, token: str) -> GrammarState:
        """
        Transition state based on observed token.
        """
        transitions = self.grammar["states"].get(state.name, [])

        # 1. Check exact matches
        for trans in transitions:
            if "token" in trans and trans["token"] == token:
                return GrammarState(trans["next"])

        # 2. Check rules
        for trans in transitions:
            if "rule" in trans:
                # Simple heuristic: if it wasn't an exact match structure token,
                # we assume it consumed the rule.
                # Note: This is a simplified DFA stepper.
                return GrammarState(trans["next"])

        # 3. No transition found -> Stuck?
        # For robustness, stay in current state or error.
        return state
