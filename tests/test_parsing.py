"""Tests du parsing : record brut -> Passage textuel."""

from sncf_agent.ingestion.parsing import _num, has_formatter, to_passage
from sncf_agent.ingestion.sources import RawRecord


def _record(dataset: str, fields: dict) -> RawRecord:
    return RawRecord(source=f"opendata:{dataset}", fields=fields, metadata={"dataset": dataset})


def test_num_arrondit_proprement():
    assert _num(10.16206349204381) == "10.2"
    assert _num(12.0) == "12"
    assert _num(19) == "19"
    assert _num(None) == ""


def test_formateur_regularite_directionnel_et_complet():
    rec = _record(
        "regularite-mensuelle-tgv-aqst",
        {
            "date": "2026-03",
            "service": "National",
            "gare_depart": "PARIS MONTPARNASSE",
            "gare_arrivee": "BORDEAUX ST JEAN",
            "duree_moyenne": 146,
            "nb_train_prevu": 1039,
            "nb_annulation": 0,
            "nb_train_depart_retard": 105,
            "retard_moyen_depart": 10.162,
            "retard_moyen_arrivee": 29.152,
            "retard_moyen_tous_trains_arrivee": 2.169,
            "nb_train_retard_sup_15": 48,
            "nb_train_retard_sup_30": 29,
            "nb_train_retard_sup_60": 4,
        },
    )
    passage = to_passage(rec)
    assert passage is not None
    txt = passage.text
    # formulation directionnelle (levier 1)
    assert "au depart de PARIS MONTPARNASSE vers BORDEAUX ST JEAN" in txt
    # champs enrichis presents
    assert "146 minutes" in txt  # duree
    assert "10.2 minutes" in txt  # retard depart arrondi
    assert "2.2 minutes" in txt  # tous trains confondus
    assert "plus de 60 minutes" in txt  # seuils


def test_formateur_regularite_rejette_record_incomplet():
    rec = _record("regularite-mensuelle-tgv-aqst", {"date": "2026-01", "gare_depart": "PARIS"})
    assert to_passage(rec) is None


def test_formateur_tarifs_avec_fourchette_de_prix():
    rec = _record(
        "tarifs-intercites",
        {
            "origine": "AGEN",
            "destination": "MONTPELLIER SAINT-ROCH",
            "transporteur": "Intercités de jour",
            "classe": "2",
            "type_place": "assise",
            "profil_tarifaire": "Tarif Normal",
            "prix_min": 19.0,
            "prix_max": 55.0,
        },
    )
    passage = to_passage(rec)
    assert passage is not None
    assert "de AGEN vers MONTPELLIER SAINT-ROCH" in passage.text
    assert "de 19 a 55 euros" in passage.text
    assert "2e classe" in passage.text


def test_repli_generique_pour_dataset_inconnu():
    rec = _record("dataset-inconnu", {"champ": "valeur", "vide": None})
    passage = to_passage(rec)
    assert passage is not None
    assert "champ: valeur" in passage.text
    assert "vide" not in passage.text  # les valeurs vides sont omises
    assert not has_formatter("dataset-inconnu")
