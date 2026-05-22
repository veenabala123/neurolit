Day 1 : 
Setup : Agent (PubMed only, Gemini Flash). Four queries run via adk web, citations checked by reading abstracts on PubMed.
Findings : 
- The agent never invented anything new or hallucinated. 
- The agent reshaped the description to fit my answer.
- Relevance stretch: Real paper, accurately described but only vaguely related to the question.
Fixes attempted :
- Description grounding step that compares the agent's draft description against the abstract and rewrites it if unsupported.
- Relevance scoring step (1–5 rubric) that demotes loose-fit papers to a "Related work" section.

Day 2 : 
Setup : Agent (PubMed + description grounding + relevance scoring + metadata-aware verifier). Reran Day 1 evaluation queries.
Findings :
- Implemented citation verification and discovered it created a new failure mode where primary sources get demoted in favor of secondary sources that lexically match the question better.
- Reviews describing the discovery scored higher than the paper that made the discovery.
- Relevance scoring is general on borderline cases.
- Agent looped on the adversarial query.
Fixes :
- Added paper metadata (title, authors, year, journal) to the relevance prompt.
- Added a hard search cap and an explicit early exit, if search returns no usable papers. 

Day 3 :
Setup: Agent adds a paper cache. When the agent looks up a paper, it saves a copy for the rest of the conversation — so if the same paper comes up again, it uses the saved copy instead of fetching it from PubMed a second time. The cache clears when the conversation ends.

Day 4 : 
Setup :  Day 4 replaces manual evaluation (running queries by hand in adk web, reading abstracts) with an automated harness. The harness drives the agent over a fixed question set via ADK's Runner API.
Findings :
- 6 question run: 100% citation precision, 0% hallucination, 2/2 adversarial passed.
- Most of the work was not the metrics, it was four measurement bugs where an infrastructure failure looked like a real result.
- A crashed run produced zero citations, which the adversarial scorer counted as a correct refusal.
- A failed PubMed network call returned the same value as a fake PMID, so a real paper got mislabeled a hallucination.
- Regex citation extraction was flaky: the same question scored differently across runs depending on how the agent formatted its citations.
Fixes :
- Crash detection so an errored run is not scored as a pass.
- Three-state PMID check (resolved / not-found / check-failed) so a network failure is excluded from metrics, not counted as a hallucination.
- Stopped parsing prose — citations are now read straight from the structured verify_and_finalize_citations tool output.

