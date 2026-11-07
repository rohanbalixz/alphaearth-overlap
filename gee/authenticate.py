"""Sign in to Earth Engine once on this machine. Opens your browser for a Google login.
(The `earthengine` command-line tool in earthengine-api 1.6.15 needs Python 3.10+; this works on 3.9.)
Run:  .venv/bin/python gee/authenticate.py
"""
import warnings

warnings.filterwarnings("ignore")
import ee  # noqa: E402

ee.Authenticate(auth_mode="localhost")
print("Signed in. Credentials saved in ~/.config/earthengine/. Next: send your Earth Engine project ID.")
