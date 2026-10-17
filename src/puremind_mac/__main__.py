"""``python -m puremind_mac`` runs the same command line as the ``puremind`` script."""

import sys

from .cli import main

sys.exit(main())
