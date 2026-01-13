# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys


class Color:
    RED = "\033[91m"
    YELLOW = "\033[93m"
    WHITE = "\033[97m"
    GREEN = "\033[92m"
    RESET = "\033[0m"


LOG_LEVELS = {"ERROR": 0, "WARN": 1, "INFO": 2, "DEBUG": 3}

COLOR_MAP = {"ERROR": Color.RED, "WARN": Color.YELLOW, "INFO": Color.WHITE, "DEBUG": Color.GREEN}


def get_env_log_level():
    level = os.environ.get("DA3_LOG_LEVEL", "INFO").upper()
    return LOG_LEVELS.get(level, LOG_LEVELS["INFO"])


class Logger:
    def __init__(self):
        self.level = get_env_log_level()

    def log(self, level_str, *args, **kwargs):
        level_key = level_str.split(":")[0].strip()
        level_val = LOG_LEVELS.get(level_key)
        if level_val is None:
            raise ValueError(f"Unknown log level: {level_str}")
        if self.level >= level_val:
            color = COLOR_MAP[level_key]
            msg = " ".join(str(arg) for arg in args)

            # Align log level output in square brackets
            # ERROR and DEBUG are 5 characters, INFO and WARN have an extra space for alignment
            tag = level_key
            if tag in ("INFO", "WARN"):
                tag += " "
            print(
                f"{color}[{tag}] {msg}{Color.RESET}",
                file=sys.stderr if level_key == "ERROR" else sys.stdout,
                **kwargs,
            )

    def error(self, *args, **kwargs):
        self.log("ERROR:", *args, **kwargs)

    def warn(self, *args, **kwargs):
        self.log("WARN:", *args, **kwargs)

    def info(self, *args, **kwargs):
        self.log("INFO:", *args, **kwargs)

    def debug(self, *args, **kwargs):
        self.log("DEBUG:", *args, **kwargs)


logger = Logger()

__all__ = ["logger"]

if __name__ == "__main__":
    logger.info("This is an info message")
    logger.warn("This is a warning message")
    logger.error("This is an error message")
    logger.debug("This is a debug message")
