# Test Questions for Document Exploration

These questions are designed to test the two-stage document exploration approach with cross-reference discovery.

## Test Scenario

**Context:** TechCorp Industries is acquiring StartupXYZ LLC. There are 10 documents in this folder related to the acquisition.

---

## Question Set 1: Simple (Single Document)

These questions can be answered from a single document:

```bash
# Q1: What is the purchase price?
explore --task "What is the total purchase price for the StartupXYZ acquisition?"

# Q2: When did the NDA get signed?
explore --task "When was the Non-Disclosure Agreement between TechCorp and StartupXYZ signed?"

# Q3: How many patents does StartupXYZ have?
explore --task "How many patents does StartupXYZ own?"
```

**Expected Behavior:**
- Agent should preview documents
- Identify the relevant document quickly
- Parse only that document for the answer

---

## Question Set 2: Medium (2-3 Documents with Cross-References)

These questions require following cross-references:

```bash
# Q4: What risks were identified and how were they addressed?
explore --task "What are the key risks identified in this acquisition and what mitigation measures were put in place?"

# Q5: What's the adjusted purchase price?
explore --task "The original purchase price was $45M. Were there any adjustments? What is the final amount?"

# Q6: What happened with customer consents?
explore --task "Which customers required consent for the acquisition, and was consent obtained from all of them?"
```

**Expected Behavior:**
- Agent previews documents
- Reads Risk Assessment Memo
- Notices references to Financial Adjustments, Customer Consents
- Follows cross-references to get complete picture

---

## Question Set 3: Complex (Multiple Documents, Deep Cross-References)

These questions require synthesizing information from many documents:

```bash
# Q7: Complete IP status
explore --task "Give me a complete picture of StartupXYZ's intellectual property - what do they own, is it properly certified, and are there any pending matters or risks?"

# Q8: Due diligence findings and resolution
explore --task "What did the due diligence process uncover, and how were any issues resolved before closing?"

# Q9: Full timeline and status
explore --task "Create a timeline of this acquisition from NDA signing to closing. What are the key milestones and their status?"

# Q10: Closing readiness
explore --task "Is this acquisition ready to close? What items are complete and what's still pending?"
```

**Expected Behavior:**
- Agent should preview all documents first
- Read the most relevant documents (e.g., Closing Checklist references everything)
- Follow cross-references to IP Certification, Due Diligence, Risk Assessment, etc.
- Synthesize information from 5+ documents

---

## Question Set 4: Adversarial (Tests Cross-Reference Discovery)

These questions specifically test if the agent goes back to previously-skipped documents:

```bash
# Q11: Following exhibit references
explore --task "The Acquisition Agreement mentions 'Exhibit A - Financial Terms'. What are the detailed financial terms?"

# Q12: Understanding document relationships  
explore --task "How does the Legal Opinion Letter relate to other documents in this acquisition?"

# Q13: Hidden connection
explore --task "Is there anything about MegaCorp in these documents? Why are they important to this deal?"
```

**Expected Behavior:**
- Q11: Agent might initially skip Financial Adjustments, but should go back when Acquisition Agreement references Exhibit A
- Q12: Agent should trace all documents referenced BY and FROM the Legal Opinion
- Q13: MegaCorp is mentioned in Due Diligence, Risk Assessment, and Customer Consents - agent should connect the dots

---

## Scoring Rubric

| Metric | Description |
|--------|-------------|
| **Preview Usage** | Did the agent use `preview_file` before `parse_file`? |
| **Selective Parsing** | Did the agent avoid parsing irrelevant documents? |
| **Cross-Reference Discovery** | Did the agent follow document references? |
| **Backtracking** | Did the agent return to previously-skipped documents when needed? |
| **Answer Completeness** | Was the final answer comprehensive and accurate? |

---

## Running a Test

```bash
export GOOGLE_API_KEY="your-key"
cd /path/to/fs-explorer
uv run explore --task "YOUR QUESTION HERE"
```

Watch for:
1. Which documents get previewed
2. Which documents get fully parsed
3. Whether the agent mentions cross-references
4. Whether the agent goes back to read referenced documents

