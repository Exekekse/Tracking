"""Dynamic rule loading and application utilities."""

import importlib
import os
import sys
import traceback
from typing import List

import config
from detection import Detection


class RuleEngine:
    """Loads rule plug‑ins from :data:`config.RULES_DIR` and applies them."""

    def __init__(self):
        self.rules = []  # list of rule instances providing ``accept``

    def load_rules(self):
        """Load or reload rule plug‑ins from the rules directory."""

        self.rules.clear()
        rules_dir = config.RULES_DIR
        pkg_name = os.path.basename(rules_dir)

        if rules_dir not in sys.path:
            sys.path.insert(0, os.path.abspath("."))

        try:
            importlib.import_module(pkg_name)
        except Exception:
            print("[RULES] Hinweis: Konnte Paket 'rules' nicht initial importieren (evtl. fehlt __init__.py).")

        if not os.path.isdir(rules_dir):
            print(f"[RULES] Ordner '{rules_dir}' nicht gefunden – keine Regeln aktiv.")
            return

        for fname in os.listdir(rules_dir):
            if not fname.endswith(".py"):
                continue
            modname = fname[:-3]
            if modname in ("__init__", "base_rule"):
                continue
            fqmn = f"{pkg_name}.{modname}"
            try:
                if fqmn in sys.modules:
                    importlib.reload(sys.modules[fqmn])
                    mod = sys.modules[fqmn]
                else:
                    mod = importlib.import_module(fqmn)
                if hasattr(mod, "build"):
                    rule = mod.build()
                    self.rules.append(rule)
                    print(f"[RULES] Geladen: {fqmn} ({getattr(rule, 'name', 'unnamed')})")
                else:
                    print(f"[RULES] Übersprungen (keine build()): {fqmn}")
            except Exception:
                print(f"[RULES] Fehler beim Laden von {fqmn}:\n{traceback.format_exc()}")

    def apply(self, dets: List[Detection], frame) -> List[Detection]:
        if not self.rules:
            return dets
        kept = []
        for d in dets:
            ok = True
            for rule in self.rules:
                try:
                    if not rule.accept(d, frame):
                        ok = False
                        break
                except Exception:
                    print(
                        f"[RULES] Fehler in Regel {getattr(rule, 'name', 'unknown')} (Detection übersprungen)"
                    )
                    ok = False
                    break
            if ok:
                kept.append(d)
        return kept

