"""Couche d'extraction : abstraction DataSource et connecteurs.

Deux implementations derriere la meme interface :

- OpenDataConnector : data.sncf.com (Opendatasoft), fonctionne SANS token (quota
  partage) ou avec une cle Opendatasoft optionnelle (quota releve). Source du prototype.
- NavitiaConnector : api.sncf.com (temps reel), necessite le token demande par
  formulaire. Stub tant que le token n'est pas arrive : le jour ou il l'est, on remplit
  fetch() et rien d'autre ne change dans le pipeline.

L'interface renvoie des RawRecord (donnee brute + provenance). Le passage
enregistrement -> texte se fait ensuite a l'etape de parsing, pas ici.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog

from sncf_agent.config import settings

log = structlog.get_logger(__name__)


class QuotaExceededError(RuntimeError):
    """Le quota d'appels anonymes data.sncf.com est atteint (errorcode 10001)."""


@dataclass(slots=True)
class RawRecord:
    """Un enregistrement brut extrait d'une source, avec sa provenance."""

    source: str  # ex. "opendata:liste-des-gares"
    fields: dict[str, Any]  # contenu brut de l'enregistrement
    metadata: dict[str, Any] = field(default_factory=dict)


class DataSource(ABC):
    """Interface commune a toutes les sources de donnees SNCF."""

    name: str

    @abstractmethod
    def fetch(self, resource: str, **kwargs: Any) -> Iterator[RawRecord]:
        """Extrait les enregistrements bruts d'une ressource (dataset, endpoint...)."""
        raise NotImplementedError


def _request_with_retry(
    client: httpx.Client,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    max_retries: int = 4,
    backoff: float = 1.5,
) -> httpx.Response:
    """GET avec retry exponentiel sur erreurs reseau et 5xx / 429."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.get(url, params=params)
            if resp.status_code in (429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError(
                    f"statut {resp.status_code}", request=resp.request, response=resp
                )
            return resp
        except httpx.HTTPError as exc:
            last_exc = exc
            wait = backoff**attempt
            log.warning(
                "requete_echouee_retry",
                url=url,
                attempt=attempt + 1,
                max_retries=max_retries,
                wait_s=round(wait, 1),
                error=str(exc),
            )
            time.sleep(wait)
    raise RuntimeError(f"echec apres {max_retries} tentatives : {url}") from last_exc


class OpenDataConnector(DataSource):
    """Connecteur data.sncf.com (Explore API v2.1, plateforme Opendatasoft).

    Utilise l'endpoint d'export pour telecharger un dataset entier en un appel.
    Fonctionne sans cle ; une cle Opendatasoft (settings.opendata_api_key) releve
    seulement le quota, elle n'est pas obligatoire.
    """

    name = "opendata"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = (base_url or settings.opendata_base_url).rstrip("/")
        self.api_key = api_key if api_key is not None else settings.opendata_api_key
        headers = {"User-Agent": "sncf-agent/0.1 (portfolio)"}
        if self.api_key:
            # Opendatasoft accepte l'authentification par header Apikey.
            headers["Authorization"] = f"Apikey {self.api_key}"
        self._client = httpx.Client(timeout=timeout, headers=headers)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OpenDataConnector:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def fetch(
        self,
        resource: str,
        *,
        select: str | None = None,
        where: str | None = None,
        order_by: str | None = None,
    ) -> Iterator[RawRecord]:
        """Telecharge tout le dataset `resource` via l'endpoint exports/json.

        resource : dataset_id data.sncf.com (ex. "liste-des-gares").
        select / where / order_by : clauses ODSQL optionnelles pour filtrer a la source.
        """
        url = f"{self.base_url}/catalog/datasets/{resource}/exports/json"
        params: dict[str, Any] = {}
        if select:
            params["select"] = select
        if where:
            params["where"] = where
        if order_by:
            params["order_by"] = order_by

        log.info("extraction_opendata", dataset=resource, has_key=bool(self.api_key))
        resp = _request_with_retry(self._client, url, params=params or None)

        # L'endpoint exports renvoie soit un tableau JSON, soit un objet d'erreur.
        payload = resp.json()
        if isinstance(payload, dict) and payload.get("errorcode") == 10001:
            raise QuotaExceededError(payload.get("error", "quota depasse"))
        resp.raise_for_status()

        if not isinstance(payload, list):
            raise RuntimeError(f"reponse export inattendue pour {resource}: {type(payload)}")

        source = f"{self.name}:{resource}"
        for rec in payload:
            yield RawRecord(source=source, fields=rec, metadata={"dataset": resource})


class NavitiaConnector(DataSource):
    """Connecteur api.sncf.com (Navitia) : itineraires et perturbations TEMPS REEL.

    STUB : necessite le token demande par formulaire (settings.sncf_api_key), encore en
    attente. Quand le token arrive : le mettre dans .env (SNCF_API_KEY) et implementer
    fetch() ci-dessous. Aucun autre changement n'est requis dans le pipeline, grace a
    l'interface DataSource commune.
    """

    name = "navitia"

    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        self.base_url = (base_url or settings.navitia_base_url).rstrip("/")
        self.api_key = api_key if api_key is not None else settings.sncf_api_key

    def fetch(self, resource: str, **kwargs: Any) -> Iterator[RawRecord]:
        if not self.api_key:
            raise NotImplementedError(
                "NavitiaConnector requiert le token api.sncf.com (SNCF_API_KEY), "
                "encore en attente. Le prototype utilise OpenDataConnector en attendant."
            )
        # TODO(token): implementer les appels Navitia (journeys, disruptions) ici,
        # avec header 'Authorization: <token>' et pagination Navitia.
        raise NotImplementedError("Integration Navitia a faire une fois le token recu.")
