import unittest
import numpy as np
from core.grammar.pruner import GrammarPruner


class TestGrammarPruner(unittest.TestCase):
    def setUp(self):
        self.pruner = GrammarPruner(grammar_name="json")

        # Mock Tokenizer
        self.vocab = {"{": 1, "}": 2, '"': 3, ":": 4, ",": 5, "true": 6}
        self.rev_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = 10

    def token_to_id(self, token):
        return self.vocab.get(token)

    def test_json_flow(self):
        # 1. Start State
        state = self.pruner.get_initial_state()
        self.assertEqual(state.name, "START")

        # 2. Prune Logits at Start
        logits = np.zeros(self.vocab_size)
        masked = self.pruner.prune_logits(logits, state, self.token_to_id)

        # Only "{" (id 1) should be unmasked
        self.assertTrue(masked[1] > -float("inf"))
        self.assertTrue(masked[2] == -float("inf"))  # "}" masked

        # 3. Transition: "{" -> OBJECT_OPEN
        state = self.pruner.update_state(state, "{")
        self.assertEqual(state.name, "OBJECT_OPEN")

        # 4. Prune at OBJECT_OPEN
        # Allowed: "}" or "\""
        masked = self.pruner.prune_logits(logits, state, self.token_to_id)
        self.assertTrue(masked[2] > -float("inf"))  # "}"
        self.assertTrue(masked[3] > -float("inf"))  # "\""
        self.assertTrue(masked[1] == -float("inf"))  # "{" invalid nested without key

        # 5. Transition: "}" -> END
        state = self.pruner.update_state(state, "}")
        self.assertEqual(state.name, "END")

    def test_value_rule(self):
        # Simulate: { "key" : <VALUE>
        state = self.pruner.get_initial_state()
        state = self.pruner.update_state(state, "{")
        state = self.pruner.update_state(state, '"')  # KEY
        state = self.pruner.update_state(
            state, "some_key"
        )  # KEY_END via ANY_STRING rule
        # Note: logic for ANY_STRING in update_state is "if no token match, take rule"

        # Let's verify that expectation
        self.assertEqual(
            state.name, "KEY"
        )  # Waait, rule loops back to KEY until quote?
        # In json.json: KEY -> ANY_STRING -> KEY.
        # So it stays in KEY until it sees a quote.

        state = self.pruner.update_state(state, '"')  # Transitions to KEY_END
        self.assertEqual(state.name, "KEY_END")

        state = self.pruner.update_state(state, ":")
        self.assertEqual(state.name, "COLON")

        # At COLON, check transitions
        # explicit: ", {, [, true, false, null
        # rule: NUMBER
        masked = self.pruner.prune_logits(np.zeros(10), state, self.token_to_id)

        # Since "NUMBER" is a rule, prune_logits sets allowed_any=True
        # So nothing should be masked?
        # Let's check impl of prune_logits
        # "if rule in trans: allowed_any = True; return logits"
        # So yes, it should allow everything (like a number or string if string was allowed)
        # In this simplified grammar COLON -> VALUE_STRING (quote) or NUMBER/literals.
        # But wait, COLON->NUMBER is a rule. So allowed_any matches.

        # Ideally we'd mask non-digits, but our pruner is "simple structure enforcement".
        self.assertTrue(np.all(masked == 0))  # None masked


if __name__ == "__main__":
    unittest.main()
