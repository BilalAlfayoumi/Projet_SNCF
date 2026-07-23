"""Tests du chunking : Passage -> Chunks avec ids stables."""

from sncf_agent.ingestion.chunking import Chunk, chunk_passages
from sncf_agent.ingestion.parsing import Passage


def test_passage_court_donne_un_seul_chunk():
    passages = [Passage(text="Un texte court.", source="test:src", metadata={"dataset": "t"})]
    chunks = chunk_passages(passages)
    assert len(chunks) == 1
    assert chunks[0].text == "Un texte court."
    assert chunks[0].source == "test:src"


def test_ids_stables_entre_deux_executions():
    passages = [Passage(text="Contenu identique.", source="test:src", metadata={})]
    a = chunk_passages(passages)
    b = chunk_passages(passages)
    assert a[0].id == b[0].id  # idempotence : meme contenu -> meme id


def test_texte_long_est_decoupe_avec_overlap():
    long_texte = ". ".join(f"Phrase numero {i} du document de test" for i in range(200))
    passages = [Passage(text=long_texte, source="test:doc", metadata={})]
    chunks = chunk_passages(passages, chunk_size=400, chunk_overlap=50)
    assert len(chunks) > 1
    # chaque chunk respecte grosso modo la taille demandee
    assert all(len(c.text) <= 400 for c in chunks)
    # ids tous distincts
    assert len({c.id for c in chunks}) == len(chunks)


def test_chunk_serialisation_aller_retour():
    c = Chunk(id="abc123", text="txt", source="s", metadata={"k": "v"})
    assert Chunk.from_dict(c.to_dict()) == c
