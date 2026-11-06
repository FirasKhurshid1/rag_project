"""
STEP 6 -- EVALUATE   (is retrieval actually finding the right chunk?)

Job: run test questions whose answers you already know. A question is a "hit"
if one of the top-k chunks comes from the expected document AND contains the
expected phrase. Add your own questions to TESTS as you go.
"""
import config
from step4_retrieve import Retriever

# (question, expected document, phrase that must appear in a retrieved chunk)
TESTS = [
    ("How many panels are planned at Site B?",               "Renewable",  "Panels planned: 150"),
    ("What is the status of Site C?",                         "Renewable",  "Equipment pending"),
    ("What safety step is recommended before work at Site C?", "Renewable",  "safety briefing"),
    ("Who is responsible for checking the taps?",             "Water",      "Maintenance team"),
    ("How did water use change from July to September?",      "Water",      "September: 86"),
    ("How often should the meter reading be recorded?",       "Water",      "meter reading"),
    ("How many people attended the online forms session?",    "Digital",    "Attended: 19"),
    ("What extra session did participants request?",          "Digital",    "file management"),
    ("What were the attendance numbers per session in the chart?", "Digital", "Intro: 22"),
    ("What challenges did the school garden face?",           "Garden",     "Hot weather"),
    ("How many plant varieties were labelled?",               "Garden",     "twelve plant varieties"),
    ("What is the status of the compost checks?",             "Garden",     "Ongoing"),
    ("How did plant growth change by week 5?",                "Garden",     "Week 5: 11"),
]


def run(top_k: int = config.TOP_K) -> float:
    r, hits_n = Retriever(), 0
    for q, doc, phrase in TESTS:
        hits = r.search(q, top_k)
        ok = any(doc.lower() in c["doc"].lower() and phrase.lower() in c["text"].lower() for _, c in hits)
        hits_n += ok
        print(f"{'PASS' if ok else 'FAIL'}  {q}")
        if not ok:
            print("      retrieved:", [f"{c['doc'][:14]}/{c['location']}" for _, c in hits])
    score = hits_n / len(TESTS)
    print(f"\nHit@{top_k}: {hits_n}/{len(TESTS)} = {score:.0%}")
    return score


if __name__ == "__main__":
    run()
