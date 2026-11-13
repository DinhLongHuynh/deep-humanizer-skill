import os
import sys


def deep_humanizer_dir():
    """Absolute path of the installed deep-humanizer skill folder (scripts/, assets/, references/)."""
    here = os.path.dirname(sys._getframe().f_code.co_filename)
    if not here or not os.path.isdir(os.path.join(here, "scripts")):
        raise RuntimeError("deep-humanizer folder not resolvable from the sidecar; locate SKILL.md's folder manually")
    return here


def deep_humanizer_cmd(script, *args):
    """Shell command string for a toolkit script, e.g. deep_humanizer_cmd('check.py', 'draft.md', '--md', 'r0.md')."""
    import shlex
    path = os.path.join(deep_humanizer_dir(), "scripts", script)
    return " ".join(["python", shlex.quote(path)] + [shlex.quote(str(a)) for a in args])
