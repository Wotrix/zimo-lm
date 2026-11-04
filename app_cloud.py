# SPDX-License-Identifier: MIT
"""Entry point for hosted demos (Streamlit Community Cloud / HF Spaces).

Training is intentionally disabled here: it needs a local CPU/GPU and a local
corpus, and long compute jobs are not appropriate on community hosting.  The
playground and the export tab work with the bundled demo model.
"""

import os

os.environ["ZIMO_CLOUD"] = "1"

from app import main  # noqa: E402  (must come after the flag is set)

main()
