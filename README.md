# Mini RAG project

Pipeline:  files -> step1 blocks -> step2 chunks -> step3 index -> step4 retrieve -> step5 answer
                                                                          step6 evaluates step4

| File | What it does |
|---|---|
| config.py | All settings |
| visuals.json | Hand-written text for charts/images (they have no extractable text) |
| step1_extract.py | Reads .docx/.pdf into neutral blocks (title, heading, para, bullet, table) |
| step2_chunk.py | Groups blocks into chunks tagged with taxonomy TYPE / SUB-TYPE |
| embedder.py | Text -> vectors (SBERT or TF-IDF fallback) |
| step3_index.py | Embeds chunks and saves index/ |
| step4_retrieve.py | Question -> top-k chunks (with optional taxonomy filters) |
| step5_answer.py | Chunks + question -> Claude answer with citations |
| step6_evaluate.py | Test questions, Hit@k score |
| main.py | Command-line entry point |

Quick start:
    pip install -r requirements.txt
    python main.py build
    python main.py ask "How many panels are planned at Site B?"
    python main.py eval
