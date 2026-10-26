"""Pinned Belebele, LongBench v2 and HotpotQA development inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from inspect_ai.dataset import Sample

from ._common import LoadedSource
from ._fetch import file_record, hf_file

BELEBELE_REVISION = "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2"
LONGBENCH_REVISION = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"
HOTPOT_REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
LANGUAGES = (
    "spa_Latn",
    "fra_Latn",
    "deu_Latn",
    "por_Latn",
    "rus_Cyrl",
    "arb_Arab",
    "hin_Deva",
    "zho_Hans",
    "jpn_Jpan",
    "swh_Latn",
)


def load_belebele(languages=None, *, data_dir: str | Path | None = None) -> LoadedSource:
    """Load parallel questions with a shared master identity across translations."""
    languages = LANGUAGES if languages is None else tuple(languages)
    if not languages or len(set(languages)) != len(languages):
        raise ValueError("Belebele needs distinct language identifiers")
    samples, files = [], []
    for language in languages:
        if not language.replace("_", "").isalnum():
            raise ValueError("invalid Belebele language")
        filename = f"data/{language}.jsonl"
        path = hf_file(
            "facebook/belebele",
            filename,
            revision=BELEBELE_REVISION,
            max_bytes=8 * 1024**2,
            data_dir=data_dir,
        )
        files.append(
            file_record(
                path, source="facebook/belebele", revision=BELEBELE_REVISION, filename=filename
            )
        )
        for line in path.read_text().splitlines():
            row = json.loads(line)
            master = json.dumps([row["link"], str(row["question_number"])], separators=(",", ":"))
            identity = hashlib.sha256(master.encode()).hexdigest()
            answer = int(row["correct_answer_num"])
            if not 1 <= answer <= 4:
                raise ValueError("invalid Belebele answer")
            samples.append(
                Sample(
                    id=f"{language}-{identity[:24]}",
                    input=f"Passage:\n{row['flores_passage']}\n\nQuestion: {row['question']}",
                    choices=[row[f"mc_answer{i}"] for i in range(1, 5)],
                    target="ABCD"[answer - 1],
                    metadata={
                        "language": language,
                        "master_question_id": identity,
                        "passage_id": row["link"],
                        "question_number": str(row["question_number"]),
                    },
                )
            )
    return LoadedSource(samples, files=files)


def load_longbench_v2(lengths=None, *, data_dir: str | Path | None = None) -> LoadedSource:
    """Retain complete source contexts; this loader never truncates inputs."""
    filename = "data.json"
    path = hf_file(
        "zai-org/LongBench-v2",
        filename,
        revision=LONGBENCH_REVISION,
        sha256="15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2",
        max_bytes=465490535,
        data_dir=data_dir,
    )
    samples = []
    for row in json.loads(path.read_text()):
        if lengths is not None and row["length"] not in lengths:
            continue
        context, question = row["context"].strip(), row["question"].strip()
        choices = "\n".join(f"({letter}) {row[f'choice_{letter}'].strip()}" for letter in "ABCD")
        prompt = (
            "Please read the following text and answer the question below.\n\n"
            f"<text>\n{context}\n</text>\n\n"
            f"What is the correct answer to this question: {question}\nChoices:\n{choices}\n\n"
            'Format your response as follows: "The correct answer is (insert answer here)".'
        )
        if row["answer"] not in "ABCD" or len(row["answer"]) != 1:
            raise ValueError("invalid LongBench answer")
        samples.append(
            Sample(
                id=row["_id"],
                input=prompt,
                target=row["answer"],
                metadata={k: row[k] for k in ("domain", "sub_domain", "difficulty", "length")},
            )
        )
    return LoadedSource(
        samples,
        files=[
            file_record(
                path, source="zai-org/LongBench-v2", revision=LONGBENCH_REVISION, filename=filename
            )
        ],
    )


def load_hotpotqa(*, data_dir: str | Path | None = None) -> LoadedSource:
    """Labeled validation distractor questions with all supplied context documents."""
    import pyarrow.parquet as pq

    filename = "distractor/validation-00000-of-00001.parquet"
    path = hf_file(
        "hotpotqa/hotpot_qa",
        filename,
        revision=HOTPOT_REVISION,
        sha256="c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6",
        max_bytes=27452575,
        data_dir=data_dir,
    )
    samples = []
    for row in pq.read_table(path).to_pylist():
        documents = []
        for title, sentences in zip(
            row["context"]["title"], row["context"]["sentences"], strict=True
        ):
            documents.append(
                f"Title: {title}\n" + "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))
            )
        facts = list(
            zip(row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"], strict=True)
        )
        prompt = (
            "Answer using the supplied documents and identify the supporting sentences. "
            'Return JSON with "answer" (a string) and "supporting_facts" '
            "(a list of [document title, zero-based sentence index] pairs).\n\n"
            + "\n\n".join(documents)
            + f"\n\nQuestion: {row['question']}"
        )
        samples.append(
            Sample(
                id=row["id"],
                input=prompt,
                target=json.dumps({"answer": row["answer"], "supporting_facts": facts}),
                metadata={
                    "type": row["type"],
                    "level": row["level"],
                    "document_titles": row["context"]["title"],
                },
            )
        )
    return LoadedSource(
        samples,
        files=[
            file_record(
                path, source="hotpotqa/hotpot_qa", revision=HOTPOT_REVISION, filename=filename
            )
        ],
    )
