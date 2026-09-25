"""Where the web layer reads the campaign from, and where it keeps its own files.

Everything the pipeline writes is only read: the campaign state database, the
ledger and the results tree. Under `data` the web layer owns web.sqlite (an
index rebuilt from those sources, safe to delete at any time), reviews.sqlite
(verdicts, the one record here that cannot be rebuilt), snippets/ (data cut
around candidates) and held/ (hard links that keep a searched beam's
filterbank until its snippets are cut).

Defaults describe the EuroFlash head node. A TOML file, ~/.config/lotaas/web.toml
unless --config names another, overrides any field by name.
"""
from dataclasses import dataclass, field, fields
from pathlib import Path
import tomllib

REPO = Path(__file__).resolve().parents[1]
LOTAAS = Path('/shared/results/dkuiper/lotaas')
DEFAULT_CONFIG = Path('~/.config/lotaas/web.toml').expanduser()


@dataclass
class Config:
    campaign_root: Path = LOTAAS / 'campaign'
    ledger: Path = LOTAAS / 'campaign.sqlite'
    data: Path = LOTAAS / 'web'
    # Cees Bassa's local archive catalogue; used for estimates, never staging.
    observation_catalogue: Path = LOTAAS / 'lotaas_observations'
    # Searched beams are found below these as .../processed/<beam>/<fingerprint>/.
    result_roots: list = field(default_factory=lambda: [LOTAAS])
    # Searched for flatfielded filterbanks of candidates found before snippets were kept.
    source_roots: list = field(default_factory=lambda: [LOTAAS])
    # Only for beams whose own metadata.json does not record the DM plan.
    settings: Path = REPO / 'settings.yaml'
    host: str = '127.0.0.1'
    port: int = 8000
    index_seconds: float = 60.0
    snippet_seconds: float = 60.0
    # Hard-link prepared filterbanks so they outlive the campaign's clean-up
    # until the snippets are cut. Costs no space until the campaign deletes its copy.
    hold: bool = True
    snippet_types: list = field(default_factory=lambda: ['candidate', 'known_pulsar'])
    # Also keep snippets of FETCH rejects scoring above this; None keeps none.
    rejected_above: float | None = None
    exclude_beams: list = field(default_factory=lambda: [12])
    # Record the verdicts the data settle (web.triage): pulses of known pulsars seen away
    # from their beam, and undispersed bursts across many beams.
    auto_triage: bool = True
    # Ask for the token cookie; off only for tests on a private machine.
    auth: bool = True
    control_dir: Path = Path('~/.ssh/control').expanduser()
    token_file: Path = Path('~/.config/lotaas/web-token').expanduser()

    @property
    def state_db(self):
        return self.campaign_root / 'campaign-state.sqlite'

    @property
    def index_db(self):
        return self.data / 'web.sqlite'

    @property
    def reviews_db(self):
        return self.data / 'reviews.sqlite'

    @property
    def snippets(self):
        return self.data / 'snippets'

    @property
    def held(self):
        return self.data / 'held'

    @property
    def logs(self):
        return self.data / 'logs'

    def prepare(self):
        for path in (self.data, self.snippets, self.held, self.logs):
            path.mkdir(parents=True, exist_ok=True)
        return self


PATHS = {'campaign_root', 'ledger', 'data', 'settings', 'control_dir', 'token_file', 'observation_catalogue'}
PATH_LISTS = {'result_roots', 'source_roots'}


def load(path=None, **overrides):
    """The defaults, then the TOML file if there is one, then explicit overrides."""
    values = {}
    path = Path(path).expanduser() if path else DEFAULT_CONFIG
    if path.is_file():
        values.update(tomllib.loads(path.read_text()))
    values.update({k: v for k, v in overrides.items() if v is not None})
    known = {f.name for f in fields(Config)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f'Unknown configuration keys: {", ".join(sorted(unknown))}')
    for name in PATHS & set(values):
        values[name] = Path(values[name]).expanduser()
    for name in PATH_LISTS & set(values):
        values[name] = [Path(p).expanduser() for p in values[name]]
    return Config(**values)
