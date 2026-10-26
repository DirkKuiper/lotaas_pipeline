"""Certificates of research-grid CAs that LTA sites' download servers present, and public trust stores lack.

Poznan's WebDAV door (webdav.lofar.psnc.pl, LT5_004's 888 archives at PSNC) presents a host certificate of the
Polish Grid CA 2019, an IGTF-accredited CA: every download from the ingest nodes failed with CERTIFICATE_VERIFY_FAILED
on 26 October 2026. The CA is trusted for those hosts only (context_for), never for the rest.
Source: IGTF distribution 1.146, ca_PolishGrid-2019-1.146.tar.gz from dist.eugridpma.info; SHA-1 fingerprint
DD:D9:1C:6A:B8:BE:B5:B3:98:F0:81:F1:F2:35:56:4E:07:07:15:09, as its .info file gives; valid to 2039-06-01.
"""
import ssl
from urllib.parse import urlsplit

POLISH_GRID_CA_2019 = """-----BEGIN CERTIFICATE-----
MIIFUjCCAzqgAwIBAgIBADANBgkqhkiG9w0BAQsFADA6MQswCQYDVQQGEwJQTDEN
MAsGA1UECgwER1JJRDEcMBoGA1UEAwwTUG9saXNoIEdyaWQgQ0EgMjAxOTAeFw0x
OTA2MDEwMDAwMDBaFw0zOTA2MDEwMDAwMDBaMDoxCzAJBgNVBAYTAlBMMQ0wCwYD
VQQKDARHUklEMRwwGgYDVQQDDBNQb2xpc2ggR3JpZCBDQSAyMDE5MIICIjANBgkq
hkiG9w0BAQEFAAOCAg8AMIICCgKCAgEAsPLrKweIHR8+YU0QO1iRASPecXnTSM+L
fT71RpBnpunrWl7LAbmRk0CUlTfXbpSU7V/kjlAmfhN40lhZhAI6XDgT/upwo+k/
XdqV/59jJokH4IlwiC0G6bw8l+XNs5PpogWdgzFWOkq/OAQfV+NqpKd8Gs7/HQDz
3OWqeg2DmEuoBEiEVNlDFXnbwA+dlfsHPHLZ+YNiSJE/Zyct6QG5PY32NQXqfHUl
9vuzVHKTiqza7kTXFMC79MPYvgWgXVy55KH5ar2LEMqHRxy+7bEaT/s1W975aToS
W0Dxpg4/dd7Kp7rtOSaBu2s0XDgoFDiOucOF0N1IGZyfSxuzFdQyFmhfC9Q6GkHx
bfIjkZZr3noOrhJZ2CpFNAfFeCvwSbTwl4t5b5huHISzv5Jlp5XRaDw1EHnL+yWT
CPp4mLDLKUVf0E7X6zEDfHXTZWuZ0uGZ2AQMYiI25QlzM5STOv9yh0q/Nb+5gaoP
iGFNP3eVyqrHVHF/7gIuKPzw3vI44aPvaozDYqpZhSDQir2t5bcFmmcqyqOFxw2K
6QZpkpit1MN9j/ShqqjTyiNtqbFIAnMohCXmaTakMHpG8DeBwT5DmGBoG/hntmvI
72pvEXq+SiILxEZb6dt3C9+t2O4duBslMMcO4iHLHs0vv/zPm2kL7DzZUNYoGlQw
zlrvq84v6EUCAwEAAaNjMGEwDwYDVR0TAQH/BAUwAwEB/zAOBgNVHQ8BAf8EBAMC
AQYwHQYDVR0OBBYEFHEO1+YMKIOjSoLgyMWXz6gJizBaMB8GA1UdIwQYMBaAFHEO
1+YMKIOjSoLgyMWXz6gJizBaMA0GCSqGSIb3DQEBCwUAA4ICAQCYxEx8DXZ6p0rx
f0hqifbjaGfEkUUhnKhE83F9Xpd9EtCvm+i/4OH6QFimwtHm3AZer7ihyarbxC/u
G3PkXAK9aHth51IBZFbfmvZAPkHJurHdIyJ3XnrfxlE/0kVtoYkxy0hQPAE5vmQW
p9J3LZq6knXKaCqgDyaJjMgDZbRDOnak6GgC34Xs1/qtamufFJp+H/HH8IRinRPi
6V3b6U2D0oWC5uTa3XwnKQvVY/QwKCp33WDTmovKqMTUdoFjzmDKAa3NeQSYe3d+
QsS1ostnGV+ZKo7oU6ND+tu/LGqfti7U8usip3/6Bitt4+geBbmF1QIG/mqf6o/N
zgZFIkxjlViKb/LxnAibMitk7hFoYVD43X14dWl/FZpjTlPMntA+PIMtOBuVtBef
0T+saCv8nV1gk17/ZMI7nF5nhFdTpNl7DFLdzEdJ6ZgYbc4Q4+DZDXB+7hxcIzfO
erjBf1ST0Luy8w+juEgokTQ/uQuFBlzABR0t5flNJPKmz0QfX81E0+hjyQg+4lUu
fvm+wPF6L4YaU/WtCEqLW+bzCxF4YCVwVGleHzp0SkXeO5Nk9KBF+JpuQXrh+2yQ
1C/rCzGEmp+OIwtNf745BteFtRS/7hzduhuIlFtJhjDp8ykQ06cjImqoi3Owt+tN
8qsUwmdaMN1V5bPznV1m0e5tj8AugQ==
-----END CERTIFICATE-----
"""

# Host suffix -> the CA certificates its servers chain to.
SITES = {'lofar.psnc.pl': (POLISH_GRID_CA_2019,)}
_contexts = {}


def context_for(url):
    """An SSL context trusting the default store and the site's grid CA, for a host of SITES; None otherwise."""
    host = urlsplit(url).hostname or ''
    for suffix, pems in SITES.items():
        if host == suffix or host.endswith('.' + suffix):
            if suffix not in _contexts:
                context = ssl.create_default_context()
                context.load_verify_locations(cadata='\n'.join(pems))
                _contexts[suffix] = context
            return _contexts[suffix]
    return None
