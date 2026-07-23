"""Parsing : transforme un RawRecord (donnee structuree brute) en Passage textuel.

Un Passage est un texte en francais, lisible et autonome, pret a etre embedde. Chaque
dataset a sa logique de formatage (ses champs sont differents), enregistree dans un
registre. Un formateur generique sert de repli pour les datasets pas encore traites.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sncf_agent.ingestion.sources import RawRecord


@dataclass(slots=True)
class Passage:
    """Un morceau de texte autonome avec ses metadonnees de provenance."""

    text: str
    source: str  # ex. "opendata:regularite-mensuelle-tgv-aqst"
    metadata: dict[str, Any] = field(default_factory=dict)


# Un formateur prend les champs bruts d'un enregistrement et renvoie un texte (ou None
# si l'enregistrement est a ignorer).
Formatter = Callable[[dict[str, Any]], str | None]

_FORMATTERS: dict[str, Formatter] = {}


def register(dataset_id: str) -> Callable[[Formatter], Formatter]:
    """Decorateur pour associer un formateur a un dataset_id."""

    def _wrap(fn: Formatter) -> Formatter:
        _FORMATTERS[dataset_id] = fn
        return fn

    return _wrap


def _clean(value: Any) -> str:
    """Normalise une valeur en texte propre (gere les espaces insecables et None)."""
    if value is None:
        return ""
    return str(value).replace("\xa0", " ").strip()


def _num(value: Any, decimals: int = 1) -> str:
    """Arrondit un nombre pour un texte lisible (10.16206349 -> '10.2', 12.0 -> '12')."""
    try:
        rounded = round(float(value), decimals)
    except (TypeError, ValueError):
        return _clean(value)
    if rounded == int(rounded):
        return str(int(rounded))
    return str(rounded)


@register("regularite-mensuelle-tgv-aqst")
def _fmt_regularite(f: dict[str, Any]) -> str | None:
    depart = _clean(f.get("gare_depart"))
    arrivee = _clean(f.get("gare_arrivee"))
    date = _clean(f.get("date"))
    if not depart or not arrivee:
        return None

    # Formulation DIRECTIONNELLE (au depart de X vers Y) : lever l'ambiguite de sens
    # que "entre X et Y" (symetrique) provoquait dans le retrieval.
    service = _clean(f.get("service"))
    entete = f"Regularite mensuelle des TGV au depart de {depart} vers {arrivee} "
    entete += f"(trajet {depart} - {arrivee}) pour la periode {date}"
    if service:
        entete += f", service {service}"
    phrases = [entete + "."]

    duree = f.get("duree_moyenne")
    if duree is not None:
        phrases.append(f"La duree moyenne du trajet est de {duree} minutes.")

    nb_prevu = f.get("nb_train_prevu")
    nb_annul = f.get("nb_annulation")
    if nb_prevu is not None:
        p = f"{nb_prevu} trains etaient prevus"
        if nb_annul is not None:
            p += f", dont {nb_annul} annules"
        phrases.append(p + ".")

    # Retards au DEPART (distincts des retards a l'arrivee)
    nb_dep = f.get("nb_train_depart_retard")
    retard_dep = f.get("retard_moyen_depart")
    if nb_dep is not None:
        p = f"Au depart : {nb_dep} trains partis en retard"
        if retard_dep is not None:
            p += f", avec un retard moyen au depart de {_num(retard_dep)} minutes"
        phrases.append(p + ".")

    # Retards a l'ARRIVEE : moyenne des trains en retard ET moyenne tous trains confondus
    retard_arr = f.get("retard_moyen_arrivee")
    if retard_arr is not None:
        phrases.append(
            f"Le retard moyen a l'arrivee des trains en retard est de {_num(retard_arr)} minutes."
        )
    retard_tous = f.get("retard_moyen_tous_trains_arrivee")
    if retard_tous is not None:
        phrases.append(
            f"Tous trains confondus (a l'heure inclus), le retard moyen a l'arrivee "
            f"est de {_num(retard_tous)} minutes."
        )

    # Repartition des retards par seuil
    seuils = []
    for champ, seuil in (
        ("nb_train_retard_sup_15", 15),
        ("nb_train_retard_sup_30", 30),
        ("nb_train_retard_sup_60", 60),
    ):
        val = f.get(champ)
        if val is not None:
            seuils.append(f"{val} trains avec plus de {seuil} minutes de retard")
    if seuils:
        phrases.append("A l'arrivee : " + ", ".join(seuils) + ".")

    # Causes de retard (pourcentages), on ne garde que celles renseignees
    causes = {
        "externe": f.get("prct_cause_externe"),
        "infrastructure": f.get("prct_cause_infra"),
        "gestion du trafic": f.get("prct_cause_gestion_trafic"),
        "materiel roulant": f.get("prct_cause_materiel_roulant"),
        "gestion en gare et reutilisation de materiel": f.get("prct_cause_gestion_gare"),
        "prise en charge des voyageurs": f.get("prct_cause_prise_en_charge_voyageurs"),
    }
    causes_txt = [f"{lib} ({_num(val)}%)" for lib, val in causes.items() if val]
    if causes_txt:
        phrases.append("Repartition des causes de retard : " + ", ".join(causes_txt) + ".")

    return " ".join(phrases)


@register("tarifs-intercites")
def _fmt_tarifs(f: dict[str, Any]) -> str | None:
    origine = _clean(f.get("origine"))
    destination = _clean(f.get("destination"))
    if not origine or not destination:
        return None
    transporteur = _clean(f.get("transporteur"))
    classe = _clean(f.get("classe"))
    profil = _clean(f.get("profil_tarifaire"))
    type_place = _clean(f.get("type_place"))
    prix_min = f.get("prix_min")
    prix_max = f.get("prix_max")

    phrase = f"Tarif Intercites pour le trajet de {origine} vers {destination}"
    details = []
    if transporteur:
        details.append(transporteur)
    if classe:
        details.append(f"{classe}e classe" if classe in ("1", "2") else f"classe {classe}")
    if type_place:
        details.append(f"place {type_place}")
    if profil:
        details.append(profil)
    if details:
        phrase += " (" + ", ".join(details) + ")"

    if prix_min is not None and prix_max is not None:
        if prix_min == prix_max:
            phrase += f" : {_num(prix_min)} euros"
        else:
            phrase += f" : de {_num(prix_min)} a {_num(prix_max)} euros"
    elif prix_min is not None:
        phrase += f" : a partir de {_num(prix_min)} euros"
    return phrase + "."


def to_passage(record: RawRecord) -> Passage | None:
    """Convertit un RawRecord en Passage via le formateur du dataset, sinon repli generique."""
    dataset = record.metadata.get("dataset", "")
    formatter = _FORMATTERS.get(dataset)
    if formatter is not None:
        text = formatter(record.fields)
    else:
        # Repli generique : "champ: valeur" pour les datasets sans formateur dedie.
        text = "; ".join(
            f"{k}: {_clean(v)}" for k, v in record.fields.items() if _clean(v)
        )
    if not text:
        return None
    return Passage(text=text, source=record.source, metadata=dict(record.metadata))


def has_formatter(dataset_id: str) -> bool:
    """Indique si un formateur dedie existe pour ce dataset."""
    return dataset_id in _FORMATTERS
