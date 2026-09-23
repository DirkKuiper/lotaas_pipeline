"""A read-mostly web layer over the LOTAAS campaign: progress, coverage and candidate review.

The pipeline never depends on anything here. This package reads the campaign
state database, the ledger and the results tree, keeps its own index and
review records under its data directory, and lives outside the folders hashed
into the run fingerprint (euroflash.run.CODE_FOLDERS), so editing it never
changes the identity of a search.
"""
