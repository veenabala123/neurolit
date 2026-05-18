Day 1 : 
Setup : Agent (PubMed only, Gemini Flash). Four queries run via adk web, citations checked by reading abstracts on PubMed.
Findings : 
- The agent never invented anything new or hallucinated. 
- The agent reshaped the description to fit my answer.
- Relevance stretch: Real paper, accurately described but only vaguely related to the question.
Fixes attempted :
- Description grounding step that compares the agent's draft description against the abstract and rewrites it if unsupported.
- Relevance scoring step (1–5 rubric) that demotes loose-fit papers to a "Related work" section.
Findings :

Day 2 : 
Setup : Agent (PubMed + description grounding + relevance scoring + metadata-aware verifier). Reran Day 1 evaluation queries.
- Implemented citation verification and discovered it created a new failure mode where primary sources get demoted in favor of secondary sources that lexically match the question better.
- Reviews describing the discovery scored higher than the paper that made the discovery.
- Relevance scoring is general on borderline cases.
Fixes :
- Added paper metadata (title, authors, year, journal) to the relevance prompt.
