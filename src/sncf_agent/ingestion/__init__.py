"""Pipeline d'ingestion : API SNCF -> parsing -> chunking -> checkpoint JSON -> embedding -> FAISS.

A implementer en Semaine 1. Le checkpoint JSON intermediaire est un choix delibere
(voir plan-projet-sncf.md) : il evite de tout relancer si l'embedding echoue.
"""
