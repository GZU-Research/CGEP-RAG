"""Plain-text versions of the extraction and answer prompts in Appendix A."""


def build_triple_prompt(question: str, reference_document: str) -> str:
    return f'''##Few-shots:##
GOOD Examples (extract these):
Example 1: "Albert Einstein was born in Ulm, Germany in 1879."
→ {{"head": "Albert Einstein", "relation": "born_in", "tail": "Ulm", "evidence": "Albert Einstein was born in Ulm, Germany in 1879."}}
→ {{"head": "Albert Einstein", "relation": "year_of_birth", "tail": "1879", "evidence": "Albert Einstein was born in Ulm, Germany in 1879."}}

Example 2: "Charles was the son of King Louis and Queen Marie."
→ {{"head": "Charles", "relation": "child_of", "tail": "King Louis", "evidence": "Charles was the son of King Louis and Queen Marie."}}
→ {{"head": "Charles", "relation": "child_of", "tail": "Queen Marie", "evidence": "Charles was the son of King Louis and Queen Marie."}}

BAD Examples (DO NOT extract these):
Example 1: Vague or incomplete triples without specific entities
→ {{"head": "He", "relation": "was", "tail": "famous"}} ← Uses pronoun instead of entity name

Example 2: Triples not supported by the text
→ {{"head": "Einstein", "relation": "invented", "tail": "computer"}} ← No textual evidence

##Instruction##
Given a question and a reference document, extract query-supporting relational triples. Each triple must be grounded in the reference document, and its original source sentence must be provided. Do not infer relations that are not supported by the reference document.

##Output Format##
Output each extracted triple as a JSON object with the fields head, relation, tail, and evidence. The evidence field must contain the original source sentence from the reference document.

{{"head": "head entity",
  "relation": "relation",
  "tail": "tail entity",
  "evidence": "source sentence"}}

##Question##
{question}

##Reference Document##
{reference_document}

##Extracted Triples##
{{triples with source sentences}}

If no valid triple can be extracted, output:
"No valid triple."'''


def build_answer_prompt(question: str, selected_evidence_documents: str) -> str:
    return f'''##Few-shots:##
{{example 1}}
Question: Who is the mother of the director of film Polish-Russian War?
Selected Evidence Documents: Polish-Russian War was directed by Xawery Żuławski. Xawery Żuławski's parents are Małgorzata Braunek and Andrzej Żuławski.
Answer: Małgorzata Braunek

{{example 2}}
Question: What is the birth year of the actor who played Harry Potter?
Selected Evidence Documents: Daniel Radcliffe played Harry Potter in the film series. Daniel Radcliffe was born in 1989.
Answer: 1989

{{...}}

##Instruction##
Given a question and a set of selected evidence documents, generate a concise answer grounded only in the provided evidence. Do not use external knowledge or infer facts that are not supported by the selected evidence.

##Question##
{question}

##Selected Evidence Documents##
{selected_evidence_documents}

##Answer##
{{answer}}

If the evidence is insufficient, output:
"Cannot be determined."'''
