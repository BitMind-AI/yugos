"""Trusted subprocess fixtures for grader supervision; no candidate code."""

import json
import os
import signal
import sys
import time
from pathlib import Path

request = json.load(sys.stdin)
Path(request["gold"]).write_text(str(os.getpid()))
if request["text"] == "crash":
    os._exit(17)
if request["text"] == "invalid":
    print("not a result")
else:
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(60)
