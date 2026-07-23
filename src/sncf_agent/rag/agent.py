"""Agent conversationnel RAG sur les donnees SNCF (LangGraph, API create_agent).

L'agent dispose d'un outil de recherche documentaire (retrieval sur l'index FAISS). Il
decide lui-meme quand chercher, formule sa reponse a partir des passages recuperes, et
cite ses sources. Construit avec create_agent (API LangChain 1.x), pas les chaines
depreciees. Le LLM par defaut est Groq (voir config), avec bascule OpenAI/Anthropic
possible via settings.llm_provider.
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import StructuredTool, tool
from langgraph.graph.state import CompiledStateGraph

from sncf_agent.config import settings
from sncf_agent.ingestion.embedding import list_indexes
from sncf_agent.rag.vectorstore import search_relevant

# Registre des corpus connus : index FAISS -> (nom d'outil, description pour le routing).
# La description est ce que lit le LLM pour choisir le bon outil : elle doit dire QUAND
# l'utiliser, pas seulement ce que c'est.
CORPUS_TOOLS: dict[str, tuple[str, str]] = {
    "regularite-mensuelle-tgv-aqst": (
        "rechercher_regularite_tgv",
        "Statistiques mensuelles de regularite des TGV par liaison : trains prevus, "
        "annulations, retards au depart et a l'arrivee (dont >15/30/60 min), duree "
        "moyenne du trajet, causes de retard. A utiliser pour toute question sur la "
        "ponctualite, les retards, les annulations ou la duree d'un trajet en TGV.",
    ),
    "tarifs-intercites": (
        "rechercher_tarifs_intercites",
        "Grilles tarifaires des trains Intercites : prix minimum et maximum par trajet "
        "(origine-destination), classe, type de place, profil tarifaire. A utiliser "
        "pour toute question de prix ou de tarif d'un billet Intercites.",
    ),
    "abonnement-max-jeune": (
        "rechercher_regles_max_jeune",
        "Regles officielles de l'abonnement MAX JEUNE (ex TGVmax, 16-27 ans) : "
        "conditions d'age, prix de l'abonnement, reservations, echanges, annulations, "
        "resiliation, trains eligibles. A utiliser pour toute question sur "
        "l'abonnement MAX JEUNE ou TGVmax.",
    ),
}

SYSTEM_PROMPT = """Tu es un assistant specialise sur les donnees de la SNCF (regularite \
des trains, tarifs, gares, correspondances). Tu reponds en francais, de maniere claire \
et concise.

PROCEDURE OBLIGATOIRE, dans cet ordre :
1. Pour toute question liee aux trains ou a la SNCF, appelle D'ABORD l'outil \
rechercher_donnees_sncf avec les mots-cles de la question. Ne reponds JAMAIS avant \
d'avoir consulte l'outil, meme si tu crois connaitre la reponse ou penses qu'il n'y \
a rien a trouver.
2. Construis ta reponse UNIQUEMENT avec les passages renvoyes : reprends leurs chiffres \
exacts et cite la source (champ source) de chaque information utilisee.
3. Seulement APRES avoir consulte l'outil : si les passages ne couvrent pas ce qui est \
demande, dis honnetement que tes donnees ne couvrent pas ce point precis, sans inventer.

Interdictions :
- Ne jamais inventer de chiffres, gares, prix, horaires ou durees.
- Ne jamais refuser de repondre sans avoir d'abord appele l'outil.
- Exception : si la question ne concerne pas du tout la SNCF ou les trains (meteo, \
restaurants...), recadre poliment vers ton domaine sans appeler l'outil."""


MULTI_SYSTEM_PROMPT = """Tu es un assistant specialise sur les donnees de la SNCF. Tu \
reponds en francais, de maniere claire et concise.

Tu disposes de PLUSIEURS outils de recherche, un par domaine (regularite des TGV, \
tarifs Intercites, regles de l'abonnement MAX JEUNE). Leurs descriptions indiquent \
quand les utiliser.

PROCEDURE OBLIGATOIRE, dans cet ordre :
1. Pour toute question liee aux trains ou a la SNCF, choisis l'outil dont le domaine \
correspond au sujet de la question et appelle-le D'ABORD. Ne reponds JAMAIS avant \
d'avoir consulte au moins un outil. Si la question touche plusieurs domaines, appelle \
plusieurs outils.
2. Construis ta reponse UNIQUEMENT avec les passages renvoyes : reprends leurs chiffres \
exacts et cite la source (champ source) de chaque information utilisee.
3. Seulement APRES avoir consulte les outils : si les passages ne couvrent pas ce qui \
est demande, dis honnetement que tes donnees ne couvrent pas ce point precis, sans \
inventer.

Interdictions :
- Ne jamais inventer de chiffres, gares, prix, horaires ou durees.
- Ne jamais refuser de repondre sans avoir d'abord appele un outil.
- Exception : si la question ne concerne pas du tout la SNCF ou les trains (meteo, \
restaurants...), recadre poliment vers ton domaine sans appeler d'outil."""


def _format_docs(docs) -> str:
    """Met en forme les passages recuperes pour le LLM."""
    if not docs:
        return "Aucun passage pertinent trouve dans ce corpus."
    blocs = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "source inconnue")
        blocs.append(f"[Passage {i} | source: {source}]\n{doc.page_content}")
    return "\n\n".join(blocs)


def _make_corpus_tool(index_name: str, tool_name: str, description: str, k: int) -> StructuredTool:
    """Cree un outil de recherche nomme pour UN corpus donne.

    La recherche est filtree par le seuil de pertinence (levier 3) : si aucun passage
    n'est assez proche, l'outil le dit explicitement et l'agent peut refuser
    honnetement au lieu de raisonner sur des passages hors-sujet.
    """

    def _search(question: str) -> str:
        return _format_docs(search_relevant(index_name, question, k=k))

    return StructuredTool.from_function(func=_search, name=tool_name, description=description)


def build_multi_agent(
    k: int | None = None,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> CompiledStateGraph:
    """Construit l'agent multi-corpus : un outil par index disponible sur disque.

    L'agent route lui-meme la question vers le bon corpus grace aux descriptions
    d'outils. Les index absents du disque sont ignores (l'agent reste utilisable
    avec les corpus effectivement ingeres).

    api_key : cle API fournie par l'utilisateur (BYOK) ; jamais stockee ni loguee,
    utilisee uniquement pour construire le client LLM de cette instance d'agent.
    """
    k = k or settings.retrieval_k
    disponibles = set(list_indexes())
    tools = [
        _make_corpus_tool(index_name, tool_name, description, k)
        for index_name, (tool_name, description) in CORPUS_TOOLS.items()
        if index_name in disponibles
    ]
    if not tools:
        raise RuntimeError("aucun index FAISS disponible : lancer les ingestions d'abord")

    effective_provider = provider or settings.llm_provider
    kwargs: dict = {"temperature": 0}
    if effective_provider == "ollama":
        kwargs["keep_alive"] = "2h"
        kwargs["num_ctx"] = 6144
        # Plafond de generation : borne la duree de redaction (~400 tokens suffisent
        # pour une reponse factuelle concise), evite les redactions interminables.
        kwargs["num_predict"] = 400
    if api_key:
        kwargs["api_key"] = api_key
    llm = init_chat_model(
        model or settings.llm_model, model_provider=effective_provider, **kwargs
    )
    return create_agent(llm, tools, system_prompt=MULTI_SYSTEM_PROMPT)


def make_retrieve_tool(index_name: str, k: int = 4):
    """Cree l'outil de recherche documentaire pour un corpus donne (avec seuil)."""

    @tool
    def rechercher_donnees_sncf(question: str) -> str:
        """Recherche des informations dans la base documentaire SNCF (regularite, tarifs,
        gares, correspondances). Fournir une question ou des mots-cles en francais."""
        return _format_docs(search_relevant(index_name, question, k=k))

    return rechercher_donnees_sncf


def build_agent(
    index_name: str,
    k: int = 4,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> CompiledStateGraph:
    """Construit l'agent RAG sur un index donne.

    index_name : nom de l'index FAISS a interroger (ex. "regularite-mensuelle-tgv-aqst").
    provider / model : override du LLM (defaut : config app, Groq). Permet d'evaluer
    l'agent avec un autre provider (ex. OpenAI) quand le quota Groq est atteint.
    """
    # temperature=0 : reponses deterministes et factuelles, et limite les derapages de
    # format d'appel d'outil des petits modeles locaux (tool call emis en texte brut).
    effective_provider = provider or settings.llm_provider
    kwargs: dict = {"temperature": 0}
    if effective_provider == "ollama":
        # keep_alive long : sans lui, Ollama decharge le modele apres ~5 min d'inactivite
        # et chaque premiere question paie ~20-30 s de rechargement GPU.
        kwargs["keep_alive"] = "2h"
        # marge de contexte pour k passages de 1400 caracteres + l'historique
        kwargs["num_ctx"] = 6144
        # Plafond de generation : borne la duree de redaction (~400 tokens suffisent
        # pour une reponse factuelle concise), evite les redactions interminables.
        kwargs["num_predict"] = 400
    if api_key:
        kwargs["api_key"] = api_key
    llm = init_chat_model(
        model or settings.llm_model,
        model_provider=effective_provider,
        **kwargs,
    )
    retrieve_tool = make_retrieve_tool(index_name, k=k)
    return create_agent(llm, [retrieve_tool], system_prompt=SYSTEM_PROMPT)


# Limite d'iterations du graphe (1 tour = LLM + outil ~ 2 pas) : borne les boucles
# d'appels d'outils repetes, couteuses avec un LLM local (~30-50 s par tour).
_RECURSION_LIMIT = 10

_MSG_LIMITE_ITERATIONS = (
    "Je n'ai pas reussi a construire une reponse complete dans le temps imparti. "
    "Essayez de reformuler ou de decouper votre question."
)


def _invoke(
    agent: CompiledStateGraph,
    question: str,
    history: list[dict] | None = None,
) -> dict:
    """Invoque l'agent avec la question, precedee de l'historique de conversation.

    Memoire courte STATELESS : l'historique vient du client a chaque requete (pas de
    checkpointer serveur). Choix delibere pour l'infra scale-to-zero : la conversation
    survit aux redemarrages de machine puisque c'est le client qui la detient. Le
    checkpointer LangGraph (thread_id + store persistant) est l'evolution prevue pour
    la version scalable.
    """
    from langgraph.errors import GraphRecursionError

    messages = [*(history or []), {"role": "user", "content": question}]
    try:
        return agent.invoke(
            {"messages": messages},
            config={"recursion_limit": _RECURSION_LIMIT},
        )
    except GraphRecursionError:
        return {"messages": []}


def ask(agent: CompiledStateGraph, question: str, history: list[dict] | None = None) -> str:
    """Pose une question a l'agent (avec l'historique eventuel) et renvoie sa reponse."""
    result = _invoke(agent, question, history=history)
    if not result["messages"]:
        return _MSG_LIMITE_ITERATIONS
    return result["messages"][-1].content


def ask_with_contexts(agent: CompiledStateGraph, question: str) -> tuple[str, list[str]]:
    """Pose une question et renvoie (reponse, contextes REELLEMENT utilises par l'agent).

    Les contextes sont extraits des messages d'outil (ToolMessage) produits par l'agent,
    donc ce sont exactement les passages qu'il a vus, pas une recherche separee. C'est
    indispensable pour une metrique de faithfulness juste. Si l'agent n'a appele aucun
    outil (ex. question hors domaine qu'il decline), la liste est vide.
    """
    result = _invoke(agent, question)
    if not result["messages"]:
        return _MSG_LIMITE_ITERATIONS, []
    messages = result["messages"]
    contexts: list[str] = []
    for msg in messages:
        # Les sorties d'outil ont le type "tool" (ToolMessage).
        if getattr(msg, "type", None) == "tool" and isinstance(msg.content, str):
            # Le retrieve tool joint les passages par "\n\n" : on les re-separe.
            contexts.extend(bloc.strip() for bloc in msg.content.split("\n\n") if bloc.strip())
    return messages[-1].content, contexts
