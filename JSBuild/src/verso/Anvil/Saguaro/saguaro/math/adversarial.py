"""Adversarial math/parser corpus generation for CPU/math verification."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AdversarialMathCase:
    """One parser-stressing case derived from a real code pattern."""

    case_id: str
    language: str
    source_text: str
    expected_expression: str | None
    should_extract: bool
    category: str

    def to_dict(self) -> dict[str, str | bool | None]:
        return {
            "case_id": self.case_id,
            "language": self.language,
            "source_text": self.source_text,
            "expected_expression": self.expected_expression,
            "should_extract": self.should_extract,
            "category": self.category,
        }


def build_adversarial_math_corpus() -> list[AdversarialMathCase]:
    """Generate a compact metamorphic corpus for math/code extraction."""

    return [
        AdversarialMathCase(
            case_id="cpp_reduction_accumulate",
            language="cpp",
            source_text=(
                "inline float reduce(float* input, int n) {\n"
                "  float acc = 0.0f;\n"
                "  for (int i = 0; i < n; ++i) {\n"
                "    acc += input[i];\n"
                "  }\n"
                "  return acc;\n"
                "}\n"
            ),
            expected_expression="acc += input[i]",
            should_extract=True,
            category="reduction",
        ),
        AdversarialMathCase(
            case_id="cpp_duplicate_clamp",
            language="cpp",
            source_text=(
                "inline float clip(float x) {\n"
                "  return std::max(0.0f, std::min(1.0f, std::max(0.0f, x)));\n"
                "}\n"
            ),
            expected_expression="return std::max(0.0f, std::min(1.0f, std::max(0.0f, x)))",
            should_extract=True,
            category="clamp",
        ),
        AdversarialMathCase(
            case_id="python_scalar_coeff",
            language="python",
            source_text=(
                "def score(x, y):\n"
                "    alpha = 0.125\n"
                "    return (x * alpha) + (y * alpha)\n"
            ),
            expected_expression="return (x * alpha) + (y * alpha)",
            should_extract=True,
            category="scalar_precompute",
        ),
        AdversarialMathCase(
            case_id="cmake_log_false_positive",
            language="yaml",
            source_text=(
                "The system is: Linux - 6.17.0-14-generic - x86_64\n"
                "Compiler: /usr/bin/c++\n"
            ),
            expected_expression=None,
            should_extract=False,
            category="build_log_noise",
        ),
        AdversarialMathCase(
            case_id="python_lambda_noise",
            language="python",
            source_text=(
                "handler = lambda token: {'width': 8, 'mode': 'safe'}\n"
                "STATE = {'lanes': 8, 'width': 32}\n"
            ),
            expected_expression=None,
            should_extract=False,
            category="config_literal_noise",
        ),
    ]
