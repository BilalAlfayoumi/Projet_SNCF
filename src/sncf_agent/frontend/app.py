"""Frontend Streamlit : chat + outil d'inspection du retrieval.

Pense comme un outil de test et d'optimisation (pas qu'une demo) :
- Onglet Chat : dialogue avec l'agent, pour juger prompt et generation.
- Onglet Inspection : passages recuperes avec scores, pour juger chunking, embedding
  et retrieval avant l'eval RAGAS systematique (Semaine 4).
- Sidebar : choix de l'index et du nombre de passages k.

Lancer le backend d'abord (uvicorn), puis ce frontend. Le frontend ne charge aucun
modele : il appelle l'API.
"""

from __future__ import annotations

# Le frontend n'importe pas la config lourde (torch...) : il appelle l'API backend.
import os

import httpx
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")
# 300 s : avec un LLM local (Ollama), une question multi-outils peut prendre 2-3 min
# (mesure LangSmith : 150 s pour decision + retrieval + redaction). Avec Groq/OpenAI,
# les reponses tombent en quelques secondes et ce timeout ne joue jamais.
TIMEOUT = 300.0

st.set_page_config(page_title="Agent RAG SNCF", page_icon="🚆", layout="wide")


@st.cache_data(ttl=30)
def fetch_indexes() -> list[str]:
    try:
        resp = httpx.get(f"{BACKEND_URL}/indexes", timeout=10.0)
        resp.raise_for_status()
        return resp.json().get("indexes", [])
    except httpx.HTTPError:
        return []


def call_chat(
    question: str,
    index: str,
    k: int,
    provider: str = "",
    api_key: str = "",
    history: list[dict] | None = None,
) -> str:
    payload: dict = {"question": question, "index": index, "k": k}
    # Memoire courte : les derniers echanges partent avec la requete (serveur stateless).
    if history:
        payload["history"] = history
    # BYOK : la cle du visiteur part avec la requete, elle n'est jamais stockee.
    if provider and api_key:
        payload["provider"] = provider.lower()
        payload["api_key"] = api_key
    resp = httpx.post(f"{BACKEND_URL}/chat", json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()["answer"]


def build_history(messages: list[dict], limite: int = 8) -> list[dict]:
    """Derniers echanges a envoyer au backend, sans les messages d'erreur."""
    propres = [
        {"role": m["role"], "content": m["content"][:4000]}
        for m in messages
        if m.get("content") and not str(m["content"]).startswith("Erreur backend")
    ]
    return propres[-limite:]


def call_search(query: str, index: str, k: int) -> list[dict]:
    resp = httpx.post(
        f"{BACKEND_URL}/search",
        json={"query": query, "index": index, "k": k},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["hits"]


st.title("🚆 Agent conversationnel SNCF")

# --- Sidebar : reglages ---
with st.sidebar:
    st.header("Reglages")
    indexes = fetch_indexes()
    if not indexes:
        st.error(
            f"Backend injoignable ou aucun index.\nBackend attendu : {BACKEND_URL}\n"
            "Lancer : uv run uvicorn sncf_agent.backend.app:app"
        )
        st.stop()
    MULTI = "multi"
    options = [MULTI, *indexes]
    index = st.selectbox(
        "Corpus interroge",
        options,
        format_func=lambda x: "Tous les corpus (agent complet)" if x == MULTI else x,
        help="En mode multi, l'agent choisit lui-meme le corpus adapte a la question.",
    )
    k = st.slider("Nombre de passages recuperes (k)", min_value=1, max_value=15, value=4)
    st.caption("k plus grand = plus de contexte, mais plus de bruit possible.")

    st.divider()
    st.subheader("Cle API personnelle")
    fournisseur = st.selectbox(
        "Fournisseur LLM",
        ["Defaut serveur", "Groq", "OpenAI"],
        help=(
            "Utilisez votre propre cle API (Groq : gratuite sur console.groq.com) : "
            "vos questions consomment alors votre quota, pas celui du serveur."
        ),
    )
    api_key = ""
    if fournisseur != "Defaut serveur":
        api_key = st.text_input(
            "Votre cle API",
            type="password",
            help="Transmise uniquement avec vos requetes, jamais stockee cote serveur.",
        )
        if not api_key:
            st.caption(":material/key: Saisissez votre cle pour activer ce fournisseur.")

tab_chat, tab_inspect = st.tabs(["Chat", "Inspection retrieval"])

# --- Onglet Chat ---
with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("Posez une question sur la SNCF..."):
        # Historique AVANT d'ajouter la nouvelle question (elle part via `question`)
        historique = build_history(st.session_state.messages)
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("L'agent reflechit..."):
                try:
                    fournisseur_actif = "" if fournisseur == "Defaut serveur" else fournisseur
                    answer = call_chat(
                        question, index, k, fournisseur_actif, api_key, historique
                    )
                except httpx.HTTPStatusError as e:
                    detail = ""
                    try:
                        detail = e.response.json().get("detail", "")
                    except Exception:  # noqa: BLE001
                        pass
                    answer = f"Erreur backend : {detail or e}"
                except httpx.HTTPError as e:
                    answer = f"Erreur backend : {e}"
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

    if st.session_state.messages and st.button("Effacer la conversation"):
        st.session_state.messages = []
        st.rerun()

# --- Onglet Inspection retrieval ---
with tab_inspect:
    st.markdown(
        "Inspecte les passages recuperes et leur score (distance L2 : **plus petit = "
        "plus proche**). Sert a juger chunking, embedding et retrieval."
    )
    if index == MULTI:
        st.info(
            "L'inspection porte sur un corpus precis : choisis un index dans la barre "
            "laterale (le mode multi ne s'applique qu'au chat)."
        )
        st.stop()
    query = st.text_input("Requete a inspecter", key="inspect_query")
    if st.button("Rechercher", key="inspect_btn") and query:
        try:
            hits = call_search(query, index, k)
        except httpx.HTTPError as e:
            st.error(f"Erreur backend : {e}")
            hits = []
        if not hits:
            st.warning("Aucun passage trouve.")
        for i, hit in enumerate(hits, 1):
            with st.container(border=True):
                cols = st.columns([1, 5])
                cols[0].metric(f"#{i}", f"{hit['score']:.3f}")
                cols[1].markdown(f"**source :** `{hit['source']}`\n\n{hit['text']}")
