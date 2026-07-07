"""Tester l'EFFICACITE du mecanisme d'attention hard-code dans le reseau.

Cadre : attention = recuperation CLE -> VALEUR par le contenu. L'etat est
[cle | valeur] ; on stocke des paires comme attracteurs ; une requete = cle
bruitee + valeur a zero ; relaxation asynchrone ; on lit la valeur obtenue.

MESURE d'efficacite = accuracy de recuperation de la valeur, comparee a
l'ATTENTION IDEALE (argmax d'overlap sur les cles = le plafond du hard). L'ecart
au ideal isole l'inefficacite du MECANISME (etats parasites) de la difficulte de
la tache (requete ambigue). On compare trois cablages : hebbien, appris a marge,
ideal argmax. Batterie : capacite (vs nb de paires), robustesse (vs bruit de
requete), latence (nb de pas).

Usage :
  python experiments/attention_efficacite.py               # tout + figure
  python experiments/attention_efficacite.py --panel P|noise ; --plot
Necessite : pip install -e . + matplotlib.
"""
import sys
import json
import argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbn_predict.regimes.associative import hebbian_weights          # noqa: E402
from sbn_predict.learning.perceptron import train_perceptron_margin  # noqa: E402
from sbn_predict.core.network import SBN                             # noqa: E402

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
NK = NV = 16
WB_L, KAPPA, SEEDS, TRIALS = 6, 10, 3, 15


def make_kv(P, nk, nv, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, (P, nk)), rng.integers(0, 2, (P, nv))


def learned_net(pats):
    n = pats.shape[1]; W = np.zeros((n, n), int); th = np.zeros(n, int)
    for j in range(n):
        w, b = train_perceptron_margin(pats, pats[:, j], wb_max=WB_L, kappa=KAPPA, epochs=200)
        W[:, j] = w; th[j] = -b
    return SBN(W=W, theta=th)


def retrieve(net, qkey, nk, nv, max_steps=200):
    x = np.concatenate([qkey, np.zeros(nv, int)]); order = list(range(nk + nv)); prev = None; steps = 0
    for _ in range(max_steps):
        x = net.step_async(x, order); steps += 1
        if prev is not None and np.array_equal(x, prev):
            break
        prev = x.copy()
    return x[nk:], steps


def flip(v, f, rng):
    if f <= 0:
        return v.copy()
    q = v.copy(); idx = rng.choice(len(v), f, replace=False); q[idx] ^= 1; return q


def eval_once(P, noise, seed):
    K, V = make_kv(P, NK, NV, seed); pats = np.concatenate([K, V], 1)
    net_h = SBN(W=hebbian_weights(pats)); net_l = learned_net(pats)
    Kpm = np.where(K > 0, 1, -1)
    oh = ol = idl = tot = 0; sh = sl = 0
    rng = np.random.default_rng(seed * 7 + 1)
    for m in range(P):
        for _ in range(TRIALS):
            q = flip(K[m], noise, rng)
            vh, sth = retrieve(net_h, q, NK, NV); vl, stl = retrieve(net_l, q, NK, NV)
            oh += int(np.array_equal(vh, V[m])); ol += int(np.array_equal(vl, V[m]))
            idl += int(int(np.argmax(Kpm @ np.where(q > 0, 1, -1))) == m)
            sh += sth; sl += stl; tot += 1
    return oh / tot, ol / tot, idl / tot, sh / tot, sl / tot


def panel_P():
    Ps = [2, 3, 4, 5, 6, 7, 8]; H = []; Lr = []; Id = []; StH = []; StL = []
    for P in Ps:
        r = np.mean([eval_once(P, 3, s) for s in range(SEEDS)], axis=0)
        H.append(r[0]); Lr.append(r[1]); Id.append(r[2]); StH.append(r[3]); StL.append(r[4])
    json.dump(dict(Ps=Ps, heb=H, learn=Lr, ideal=Id, stepsH=StH, stepsL=StL),
              open(FIG_DIR / "attn_P.json", "w"))
    print("P:", list(zip(Ps, [round(x, 2) for x in H], [round(x, 2) for x in Lr], [round(x, 2) for x in Id])))


def panel_noise():
    P = 5; noises = [0, 1, 2, 3, 4, 6, 8]; H = []; Lr = []; Id = []
    for nz in noises:
        r = np.mean([eval_once(P, nz, s) for s in range(SEEDS)], axis=0)
        H.append(r[0]); Lr.append(r[1]); Id.append(r[2])
    json.dump(dict(noises=noises, P=P, heb=H, learn=Lr, ideal=Id), open(FIG_DIR / "attn_noise.json", "w"))
    print("noise:", list(zip(noises, [round(x, 2) for x in H], [round(x, 2) for x in Lr], [round(x, 2) for x in Id])))


def plot():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dP = json.load(open(FIG_DIR / "attn_P.json")); dN = json.load(open(FIG_DIR / "attn_noise.json"))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    a = ax[0]
    a.plot(dP["Ps"], dP["ideal"], "^-", c="#27ae60", label="attention IDEALE (argmax)")
    a.plot(dP["Ps"], dP["learn"], "o-", c="#2471a3", label="reseau appris a marge")
    a.plot(dP["Ps"], dP["heb"], "s--", c="#c0392b", label="reseau hebbien")
    a.plot(dP["Ps"], [1.0 / P for P in dP["Ps"]], ":", c="grey", label="hasard (1/P)")
    a.set_xlabel("nb de paires cle-valeur stockees"); a.set_ylabel("recuperation valeur correcte")
    a.set_title("CAPACITE d'attention (bruit requete = 3 bits)\nl'ecart au ideal = efficacite du mecanisme")
    a.legend(fontsize=8); a.set_ylim(0, 1.05)
    b = ax[1]
    b.plot(dN["noises"], dN["ideal"], "^-", c="#27ae60", label="ideale (argmax)")
    b.plot(dN["noises"], dN["learn"], "o-", c="#2471a3", label="appris a marge")
    b.plot(dN["noises"], dN["heb"], "s--", c="#c0392b", label="hebbien")
    b.set_xlabel(f"bruit sur la requete (bits, /{NK})"); b.set_ylabel("recuperation valeur correcte")
    b.set_title(f"ROBUSTESSE au bruit de requete (P={dN['P']})"); b.legend(fontsize=8); b.set_ylim(0, 1.05)
    c = ax[2]
    c.plot(dP["Ps"], dP["stepsH"], "s--", c="#c0392b", label="hebbien")
    c.plot(dP["Ps"], dP["stepsL"], "o-", c="#2471a3", label="appris a marge")
    c.set_xlabel("nb de paires"); c.set_ylabel("pas de convergence (async)")
    c.set_title("LATENCE (efficacite en temps)"); c.legend(fontsize=8)
    fig.suptitle("Efficacite du mecanisme d'attention hard-code : recuperation cle->valeur vs l'attention "
                 "ideale (argmax)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = FIG_DIR / "attention_efficacite.png"; fig.savefig(out, dpi=130); plt.close(fig); print("Figure:", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", choices=["P", "noise"])
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args(); FIG_DIR.mkdir(exist_ok=True)
    if a.plot:
        plot()
    elif a.panel == "P":
        panel_P()
    elif a.panel == "noise":
        panel_noise()
    else:
        panel_P(); panel_noise(); plot()


if __name__ == "__main__":
    main()
