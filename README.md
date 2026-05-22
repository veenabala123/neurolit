Neurolit
An agentic literature review assistant for researchers entering neuroscience from adjacent fields. Built with Google ADK and Gemini. 
NeuroLit searches PubMed, retrieves real papers, verifies that each citation's description is supported by the paper's abstract, scores how directly each paper answers the user's question, and produces a structured synthesis grounded in primary sources rather than the LLM's training-data priors.

```
neurolit/
│
├── neurolit_agent/             ← the agent itself
│   ├── agent.py                  define the agent + its tools
│   ├── verification.py           grounding + relevance checks
│   ├── schemas.py                Pydantic models for structured output
│   ├── retry.py                  retry on Gemini 429 / 503 errors
│   ├── __init__.py
│   └── .env.example              copy to .env, add your Gemini key
├── eval/                       Automated evaluation harness
│   ├── questions.py              the eval question set
│   ├── scorers.py                deterministic scorers
│   ├── run_eval.py               runs the agent, scores, writes a report
│   └── __init__.py
│
├── notes/                      ← evaluation diary, drives each iteration
│   ├── project_notes.md      
│
├── requirements.txt
└── README.md

```

## How to run

You'll need Python 3.10+ and a free Gemini API key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

```bash
git clone https://github.com/YOUR-USERNAME/neurolit.git
cd neurolit

python -m venv .venv
source .venv/bin/activate         
pip install -r requirements.txt

cp neurolit_agent/.env.example neurolit_agent/.env
# open neurolit_agent/.env and paste your key after GOOGLE_API_KEY=

adk web
```

Open [http://localhost:8000](http://localhost:8000), pick `neurolit_agent` from the dropdown, and ask it something.