"""Jalon 3 (synthese §6) — Apprentissage ASP.

Ce script génère une instance .lp à partir d'un petit jeu (x -> y) puis appelle
clingo si disponible. Mesure le mur de scalabilité (n, taille du jeu).

Prérequis : clingo installé (https://potassco.org). python experiments/milestone3_asp.py
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ASP_DIR = Path(__file__).resolve().parents[1] / "asp"


def main() -> None:
    clingo = shutil.which("clingo")
    if clingo is None:
        print("clingo introuvable — installe-le (conda install -c potassco clingo).")
        print(f"Sinon, lance manuellement : clingo {ASP_DIR/'learn_weights.lp'} "
              f"{ASP_DIR/'instance_example.lp'}")
        return
    res = subprocess.run(
        [clingo, str(ASP_DIR / "learn_weights.lp"), str(ASP_DIR / "instance_example.lp")],
        capture_output=True, text=True,
    )
    print(res.stdout)


if __name__ == "__main__":
    main()
