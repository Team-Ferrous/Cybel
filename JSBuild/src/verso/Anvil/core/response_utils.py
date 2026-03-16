import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def clean_response(text: str) -> str:
    """
    Final cleanup of text for Anvil models.
    Strips special tokens, redundant role labels, and skeleton-like artifacts.

    PERFORMANCE FIX: Comprehensive cleaning to remove garbage output.
    """
    if not text:
        return ""

    # 0. Pre-clean: Ensure text is a string
    if not isinstance(text, str):
        text = str(text)

    # 1. Remove complete role blocks: <|start_of_role|>...<|end_of_role|>
    text = re.sub(r"<\|start_of_role\|>.*?<\|end_of_role\|>", "", text, flags=re.DOTALL)

    # 2. Remove all variants of Anvil tags EXCEPT tool-related ones
    text = re.sub(r"<\|(?!tool_call|/tool_call)[a-z_]*\|>", "", text)
    text = re.sub(r"\|[a-z_]*\|>", "", text)
    text = re.sub(r"<\|(?!tool_call)[a-z_]*", "", text)

    # 3. Clean up leaked role labels at start of text or lines
    text = re.sub(
        r"^(assistant|user|system):\s*", "", text, flags=re.MULTILINE | re.IGNORECASE
    )
    text = re.sub(
        r"^(assistant|user|system)\s*\n", "", text, flags=re.MULTILINE | re.IGNORECASE
    )

    # 4. Remove garbage model tokens (comprehensive list)
    garbage_patterns = [
        r"<\|start_of_text\|>",
        r"<\|end_of_text\|>",
        r"<\|end_of_context\|>",
        r"<\|start_of_text\|\|end of context",
        r"<\|end of conversation\|\|>",
        r"\|end of conversation\|",
        r"<\|endoftext\|>",
        r"<\|im_start\|>",
        r"<\|im_end\|>",
        r"<s>",
        r"</s>",
        r"\[INST\]",
        r"\[/INST\]",
        r"<<SYS>>",
        r"<</SYS>>",
        # Incomplete/malformed tokens
        r"<\|end_text\|?",
        r"<\|assistant\|?>",
        r"<\|user\|?>",
        r"<\|system\|?>",
        r"<\|end_of_response\|?>",
        r"<\|--\s*end_of_response\s*\|-->",
        r"<\|endOf_response\s*\|-->",
        # COCONUT thinking tokens
        r"<\|end_of_thinking\|?>",
        r"\|end_of_thinking\|",
        r"<\|start_of_thinking\|?>",
        r"\|start_of_thinking\|",
        # Markdown artifacts
        r"```<\|",
        r"```markdown\n#.*?\n```",
        r"`{3,}<\|",
        r"```markdown\n```",
        r"(?:assistant|user)```",
        r"```\n\n```",
        # Repeated headers (e.g., "## Summary\n## Summary")
        r"(^##\s+[^\n]+\n)\1+",
        # Empty code blocks
        r"```\s*```",
        r"```python\s*```",
        r"```\n\s*\n```",
    ]
    for pattern in garbage_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)

    # 4b. Remove leaked internal tool-call payloads and loop-recovery markers
    text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL)
    text = re.sub(
        r'^\s*\{\s*"name"\s*:\s*"[a-zA-Z0-9_]+"\s*,\s*"arguments"\s*:\s*\{.*\}\s*\}\s*$',
        "",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(r"\[SYSTEM:\s*Streaming loop terminated\.\]", "", text)
    text = re.sub(r"\[SYSTEM:\s*Thinking loop terminated\.\]", "", text)

    # 5. Clean up any remaining <| or |> fragments
    text = re.sub(r"<\|[^>]*$", "", text)  # Trailing incomplete tags
    text = re.sub(r"\|\|>", "", text)
    text = re.sub(r"\|>", "", text)

    # 6. Remove excessive whitespace
    text = re.sub(r"\n{4,}", "\n\n\n", text)  # Max 3 consecutive newlines
    return text.strip()


def finalize_synthesis(
    raw_response: str, thinking_system, brain, history: List[Dict[str, Any]], console
) -> str:
    """
    Finalizes the synthesis process by separating thoughts from the answer.
    If the answer is empty, it uses the thoughts to generate a new answer.
    """
    logger.debug("Finalizing synthesis...")

    # 1. Parse thinking blocks from the raw response
    thinking_blocks = thinking_system.parser.parse(raw_response)
    if thinking_blocks:
        logger.info(f"Parsed {len(thinking_blocks)} thinking blocks from response.")
        # Add to the official thinking chain for this task
        if thinking_system.current_chain is not None:
            thinking_system.current_chain.blocks.extend(thinking_blocks)

    # 2. Get the clean answer by removing the thinking blocks
    clean_answer = thinking_system.parser.remove_thinking_blocks(raw_response)
    logger.debug(f"Clean answer after removing blocks: '{clean_answer[:100]}...'")

    # 3. Check if the answer is empty or just whitespace
    if not clean_answer.strip():
        logger.warning("Main answer is empty. Synthesizing answer from thoughts.")
        console.print(
            "[dim]Main answer was empty, synthesizing a final response from the agent's thoughts...[/dim]"
        )

        # Fallback: If there are no thoughts, return a default message
        if not thinking_blocks:
            logger.error("No answer and no thoughts found. Returning error message.")
            return "I have analyzed the request but could not formulate a final answer."

        # Collate the thoughts into a single block of text
        thoughts_text = "\n\n".join([block.to_xml() for block in thinking_blocks])

        # Create a new prompt to synthesize the thoughts
        synthesis_prompt = f"""You are a final response synthesizer. Your task is to convert the following internal monologue (in <thinking> blocks) into a clear, concise, and user-facing answer. Do not add any new information. Just present the conclusions from the thoughts.

Internal Monologue:
{thoughts_text}

Synthesize the final user-facing answer based on these thoughts. Do not include any <thinking> blocks in your output.
"""
        messages = [
            {
                "role": "system",
                "content": "You are an expert synthesizer. You create clean, final answers from internal reasoning steps.",
            },
            {"role": "user", "content": synthesis_prompt},
        ]

        # Use the brain to generate the final answer (non-streamed)
        final_answer = ""
        try:
            # Assuming brain has a simple chat method that returns a string
            # If it only streams, we'd accumulate the chunks here.
            response_stream = brain.stream_chat(
                messages, max_tokens=4096, temperature=0.1
            )
            for chunk in response_stream:
                final_answer += chunk
        except Exception as e:
            logger.error(f"Error during fallback synthesis: {e}")
            final_answer = "I apologize, but I encountered an error while trying to synthesize my thoughts into a final answer."

        return final_answer.strip()

    return clean_answer


class ResponseStreamEvent:
    def __init__(self, type: str, content: str = "", metadata: dict = None):
        self.type = type  # 'content', 'thinking_start', 'thinking_chunk', 'thinking_end', 'tool_start', 'tool_chunk', 'tool_end'
        self.content = content
        self.metadata = metadata or {}


class ResponseStreamParser:
    """
    State-aware parser for Anvil model streams.
    Separates natural language, structured thinking, and tool calls in real-time.
    Supports standard XML tags, Anvil's custom tool delimiters, and XML-attribute formats.
    """

    def __init__(self):
        self.buffer = ""
        self.state = "content"  # 'content', 'thinking', 'tool_call'
        self.current_tag = ""
        self.thinking_type = "reasoning"
        self.current_think_tag = "thinking"

        # Tags to hide/capture
        self.THINK_START = re.compile(
            r'<(thinking|reasoning|thought)(?: type=["\'](\w+)["\'])?\s*>'
        )
        self.THINK_END_TAGS = ["</thinking>", "</reasoning>", "</thought>"]

        # Tool Call Delimiters
        self.TOOL_START_TAG = "<tool_call>"
        self.TOOL_END_TAG = "</tool_call>"
        self.TOOL_START_CUSTOM = "tool|>"
        self.TOOL_END_CUSTOM = "<|"

        # Model-specific variations (leaked or alternate formats)
        self.TOOL_START_XML_ATTR = re.compile(r'tool<tool_name=["\'](\w+)["\']')
        self.TOOL_END_XML_ATTR = "</tool>"

        self.ROLE_TAG = re.compile(
            r"<\|.*?\|>|(?:\b[Aa]ssistant|\b[Uu]ser|\b[Ss]ystem)\s*[:|>]", re.IGNORECASE
        )
        self.JUNK_TAG = re.compile(
            r"assistant\s*\|?|user\s*\|?|system\s*\|?", re.IGNORECASE
        )

    def process_chunk(self, chunk: str):
        """Processes a chunk and yields stream events."""
        self.buffer += chunk

        while self.buffer:
            if self.state == "content":
                # Check for starts of transitions
                think_match = self.THINK_START.search(self.buffer)
                tool_xml_attr_match = self.TOOL_START_XML_ATTR.search(self.buffer)

                tool_start_tag_idx = self.buffer.find(self.TOOL_START_TAG)
                tool_start_custom_idx = self.buffer.find(self.TOOL_START_CUSTOM)

                role_match = self.ROLE_TAG.search(self.buffer)
                tag_start_idx = self.buffer.find("<")

                # Find the earliest transition
                indices = []
                if think_match:
                    indices.append((think_match.start(), "thinking"))
                if tool_xml_attr_match:
                    indices.append((tool_xml_attr_match.start(), "tool_xml_attr"))
                if tool_start_tag_idx != -1:
                    indices.append((tool_start_tag_idx, "tool_tag"))
                if tool_start_custom_idx != -1:
                    indices.append((tool_start_custom_idx, "tool_custom"))
                if role_match:
                    indices.append((role_match.start(), "role"))

                if not indices:
                    # Look for partial triggers
                    lookback_check = False
                    if tag_start_idx != -1:
                        if tag_start_idx > 0:
                            yield ResponseStreamEvent(
                                "content", self.buffer[:tag_start_idx]
                            )
                            self.buffer = self.buffer[tag_start_idx:]
                        lookback_check = True

                    if not lookback_check:
                        # Check for partial "tool"
                        p_idx = self.buffer.find("tool")
                        if p_idx != -1:
                            if p_idx > 0:
                                yield ResponseStreamEvent(
                                    "content", self.buffer[:p_idx]
                                )
                                self.buffer = self.buffer[p_idx:]
                            lookback_check = True

                    if not lookback_check:
                        if len(self.buffer) > 15:
                            yield ResponseStreamEvent("content", self.buffer[:-15])
                            self.buffer = self.buffer[-15:]
                        break
                    else:
                        break
                else:
                    indices.sort()
                    min_idx, next_state = indices[0]

                    if min_idx > 0:
                        yield ResponseStreamEvent("content", self.buffer[:min_idx])
                        self.buffer = self.buffer[min_idx:]

                    if next_state == "thinking":
                        match = self.THINK_START.match(self.buffer)
                        if match:
                            self.thinking_type = (
                                match.group(2) or match.group(1) or "reasoning"
                            )
                            self.state = "thinking"
                            self.current_think_tag = match.group(1)
                            self.buffer = self.buffer[match.end() :]
                            logger.debug(
                                f"Detected thinking start: {self.thinking_type} via <{self.current_think_tag}>"
                            )
                            yield ResponseStreamEvent(
                                "thinking_start",
                                metadata={
                                    "type": self.thinking_type,
                                    "tag": self.current_think_tag,
                                },
                            )
                        else:
                            break
                    elif next_state == "tool_xml_attr":
                        self.state = "tool_call"
                        self.current_tag = "xml_attr"
                        # We don't advance past the start tag here because we need it for extraction logic
                        # but we emit tool_start to stop streaming
                        yield ResponseStreamEvent("tool_start")
                        # Advance buffer slightly to prevent immediate re-match if necessary
                        # but keeping the 'tool<...' helps _extract_tool_calls
                        # Wait, _extract_tool_calls works on full_response.
                        # To stop the stream, we just need to yield tool_start and change state.
                        return  # Stop processing this turn if we found a tool start
                    elif next_state == "tool_tag":
                        self.state = "tool_call"
                        self.current_tag = "tag"
                        self.buffer = self.buffer[len(self.TOOL_START_TAG) :]
                        yield ResponseStreamEvent("tool_start")
                    elif next_state == "tool_custom":
                        self.state = "tool_call"
                        self.current_tag = "custom"
                        self.buffer = self.buffer[len(self.TOOL_START_CUSTOM) :]
                        yield ResponseStreamEvent("tool_start")
                    elif next_state == "role":
                        match = self.ROLE_TAG.match(self.buffer)
                        if match:
                            logger.debug(f"Stripped role tag: {repr(match.group(0))}")
                            self.buffer = self.buffer[match.end() :]
                        else:
                            break

            elif self.state == "thinking":
                # Find which end tag matches the current thinking type
                end_tag = f"</{self.current_think_tag}>"
                idx = self.buffer.find(end_tag)
                if idx == -1:
                    # Generic fallback check for any thinking end if the specific one is missed
                    for fallback_tag in self.THINK_END_TAGS:
                        if fallback_tag in self.buffer:
                            idx = self.buffer.find(fallback_tag)
                            end_tag = fallback_tag
                            break

                if idx == -1:
                    # Look for ANY tool start signal during thinking as a force-exit
                    if any(
                        s in self.buffer
                        for s in [self.TOOL_START_TAG, self.TOOL_START_CUSTOM, "tool<"]
                    ):
                        # Find earliest
                        indices = [
                            self.buffer.find(s)
                            for s in [
                                self.TOOL_START_TAG,
                                self.TOOL_START_CUSTOM,
                                "tool<",
                            ]
                            if s in self.buffer
                        ]
                        split_idx = min(indices)
                        yield ResponseStreamEvent(
                            "thinking_chunk", self.buffer[:split_idx]
                        )
                        self.buffer = self.buffer[split_idx:]
                        self.state = "content"
                        yield ResponseStreamEvent("thinking_end")
                    elif len(self.buffer) > 2000:  # Emergency split
                        yield ResponseStreamEvent("thinking_chunk", self.buffer[:1000])
                        self.buffer = self.buffer[1000:]
                    break
                else:
                    yield ResponseStreamEvent("thinking_chunk", self.buffer[:idx])
                    self.buffer = self.buffer[idx + len(end_tag) :]
                    self.state = "content"
                    yield ResponseStreamEvent("thinking_end")

            elif self.state == "tool_call":
                # In tool_call state, we look for the end marker.
                # If we don't find it but see a new thinking start or too much content, force stop.
                end_marker = {
                    "tag": self.TOOL_END_TAG,
                    "custom": self.TOOL_END_CUSTOM,
                    "xml_attr": self.TOOL_END_XML_ATTR,
                }.get(self.current_tag, self.TOOL_END_TAG)

                idx = self.buffer.find(end_marker)
                if idx == -1:
                    # Force stop if we see tokens like <|start_of_role|> or another thinking block
                    if (
                        self.ROLE_TAG.search(self.buffer)
                        or self.THINK_START.search(self.buffer)
                        or len(self.buffer) > 3000
                    ):
                        yield ResponseStreamEvent("tool_chunk", self.buffer)
                        self.buffer = ""
                        self.state = "content"
                        yield ResponseStreamEvent("tool_end")
                    break
                else:
                    yield ResponseStreamEvent("tool_chunk", self.buffer[:idx])
                    self.buffer = self.buffer[idx + len(end_marker) :]
                    self.state = "content"
                    yield ResponseStreamEvent("tool_end")

    def finalize(self):
        """Flush remaining buffer."""
        if self.buffer:
            if self.state == "thinking":
                yield ResponseStreamEvent("thinking_chunk", self.buffer)
                yield ResponseStreamEvent("thinking_end")
            elif self.state == "tool_call":
                yield ResponseStreamEvent("tool_chunk", self.buffer)
                yield ResponseStreamEvent("tool_end")
            else:
                cleaned = self.buffer
                cleaned = re.sub(r"<\|.*?\|>", "", cleaned)
                if cleaned:
                    yield ResponseStreamEvent("content", cleaned)
        self.buffer = ""
