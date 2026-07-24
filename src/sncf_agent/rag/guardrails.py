"""Guardrails d'entree et de sortie de l'agent RAG.

Deux familles, deux moments :

- ENTREE (avant le LLM) : bornage de taille (fait par pydantic dans le backend),
  detection d'injection de prompt, et detection de donnees personnelles (PII). Une
  question contenant du PII est bloquee AVANT d'atteindre le LLM (qui peut etre un
  provider externe) : la donnee n'est ni envoyee, ni loguee.
- SORTIE (apres le LLM) : verification que les sources citees dans la reponse existent
  bien dans les passages reellement recuperes. Une citation absente des contextes est
  probablement inventee : on la signale plutot que de la laisser passer en silence.

Regles deterministes (regex + Luhn), sans appel LLM : rapides, previsibles, testables.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --- Injection de prompt (entree) ---
_INJECTION_RE = re.compile(
    r"(ignore\s+(les|tes|toutes)\s+.{0,20}instructions"
    r"|oublie\s+(les|tes)\s+.{0,20}instructions"
    r"|system\s*prompt|prompt\s*syst[eè]me"
    r"|tu\s+es\s+maintenant\s+"
    r"|reveal\s+your\s+instructions)",
    re.IGNORECASE,
)

# --- PII (entree) ---
# Patterns cibles pour minimiser les faux positifs (un prix ou un numero de train ne
# doit pas etre pris pour du PII). La carte bancaire est confirmee par un test de Luhn.
_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b"),
    "telephone": re.compile(r"\b0[1-9](?:[ .\-]?\d{2}){4}\b"),  # numero francais
    "iban": re.compile(r"\bFR\d{2}(?:[ ]?[A-Z0-9]{4}){5}(?:[ ]?[A-Z0-9]{1,3})\b", re.IGNORECASE),
    # Numero de securite sociale FR : 15 chiffres exactement (13 + cle), cle obligatoire
    # pour eviter de confondre avec un numero de train ou une reference a 13 chiffres.
    "numero_securite_sociale": re.compile(
        r"\b[12][ ]?\d{2}[ ]?\d{2}[ ]?\d{2}[ ]?\d{3}[ ]?\d{3}[ ]?\d{2}\b"
    ),
}
# Suite de 13 a 19 chiffres (avec espaces/tirets) : candidate carte bancaire, validee ensuite.
_CARTE_CANDIDATE_RE = re.compile(r"\b(?:\d[ \-]?){13,19}\b")


def _luhn_ok(digits: str) -> bool:
    """Test de Luhn : filtre les suites de chiffres qui ne sont pas des cartes valides."""
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def detect_pii(text: str) -> list[str]:
    """Renvoie les TYPES de PII detectes (jamais les valeurs), sans doublon, tries."""
    trouves: set[str] = set()
    for type_pii, pattern in _PII_PATTERNS.items():
        if pattern.search(text):
            trouves.add(type_pii)
    for candidat in _CARTE_CANDIDATE_RE.findall(text):
        chiffres = re.sub(r"\D", "", candidat)
        if 13 <= len(chiffres) <= 19 and _luhn_ok(chiffres):
            trouves.add("carte_bancaire")
            break
    return sorted(trouves)


def detect_injection(text: str) -> bool:
    """Vrai si la question ressemble a une tentative d'injection de prompt."""
    return bool(_INJECTION_RE.search(text))


@dataclass
class InputVerdict:
    """Resultat du controle d'entree. blocked=True -> ne pas appeler le LLM."""

    blocked: bool
    reason: str = ""  # "injection" | "pii" | ""
    message: str = ""  # message a renvoyer a l'utilisateur si bloque
    pii_types: list[str] = field(default_factory=list)  # types seulement, jamais les valeurs


_MSG_INJECTION = (
    "Je ne peux pas traiter cette demande. Posez-moi une question sur les donnees "
    "SNCF : regularite des TGV, tarifs Intercites ou abonnement MAX JEUNE."
)
_MSG_PII = (
    "Votre message semble contenir des donnees personnelles (email, telephone, "
    "coordonnees bancaires...). Pour votre securite, ne partagez pas d'informations "
    "personnelles : reformulez votre question sans ces details."
)


def scan_input(question: str) -> InputVerdict:
    """Controle d'entree complet : injection puis PII. Bloque au premier motif trouve."""
    if detect_injection(question):
        return InputVerdict(blocked=True, reason="injection", message=_MSG_INJECTION)
    pii = detect_pii(question)
    if pii:
        return InputVerdict(blocked=True, reason="pii", message=_MSG_PII, pii_types=pii)
    return InputVerdict(blocked=False)


# --- Verification des sources citees (sortie) ---
# On extrait les sources REELLEMENT recuperees depuis les contextes (format produit par
# _format_docs : "[Passage i | source: X]"), et les URLs citees dans la reponse. Une URL
# de la reponse absente des sources recuperees est une citation non verifiee.
_SOURCE_DANS_CONTEXTE_RE = re.compile(r"source:\s*([^\]]+?)\s*\]")
_URL_RE = re.compile(r"https?://[^\s)\]}>,\"']+")


def _sources_autorisees(contexts: list[str]) -> set[str]:
    autorisees: set[str] = set()
    for bloc in contexts:
        for m in _SOURCE_DANS_CONTEXTE_RE.findall(bloc):
            autorisees.add(m.strip())
    return autorisees


@dataclass
class OutputVerdict:
    """Resultat du controle de sortie. unsupported = URLs citees hors des contextes."""

    unsupported: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.unsupported


def verify_sources(answer: str, contexts: list[str]) -> OutputVerdict:
    """Verifie que les URLs citees dans la reponse figurent dans les passages recuperes.

    Renvoie les URLs citees qui n'apparaissent dans aucune source recuperee (citations
    non ancrees, potentiellement inventees). Une reponse sans URL est consideree OK
    (l'agent cite souvent le champ source en clair, gere par le grounding RAGAS ailleurs).
    """
    urls_citees = _URL_RE.findall(answer)
    if not urls_citees:
        return OutputVerdict()
    autorisees = _sources_autorisees(contexts)
    corpus_sources = " ".join(autorisees)
    unsupported = [
        url for url in dict.fromkeys(urls_citees)  # dedoublonne en gardant l'ordre
        if url.rstrip("/.") not in corpus_sources
    ]
    return OutputVerdict(unsupported=unsupported)
