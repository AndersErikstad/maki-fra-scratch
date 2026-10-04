import math
import random
import sys


def datasett(navn):
    hjørner = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    fasit = {"xor": [0, 1, 1, 0], "and": [0, 0, 0, 1], "or": [0, 1, 1, 1]}
    if navn in fasit:
        return [(x, y, t) for (x, y), t in zip(hjørner, fasit[navn])]
    if navn == "sirkel":
        punkter = [(i / 4, j / 4) for i in range(-4, 5) for j in range(-4, 5)]
        return [(x, y, int(x * x + y * y < 0.58**2)) for x, y in punkter]
    raise ValueError("Velg xor, and, or eller sirkel")


class Nettverk:
    def __init__(self, seed=9173, nevroner=8):
        r = random.Random(seed)
        self.w1 = [[r.uniform(-1, 1), r.uniform(-1, 1)] for _ in range(nevroner)]
        self.b1 = [r.uniform(-0.2, 0.2) for _ in range(nevroner)]
        self.w2 = [r.uniform(-1, 1) for _ in range(nevroner)]
        self.b2 = 0.0

    def forward(self, x, y):
        h = [math.tanh(wx * x + wy * y + b) for (wx, wy), b in zip(self.w1, self.b1)]
        z = sum(w * a for w, a in zip(self.w2, h)) + self.b2
        e = math.exp(-abs(z))
        p = 1 / (1 + e) if z >= 0 else e / (1 + e)
        return h, z, p

    def loss(self, data):
        # BCE fra logiten, så vi slipper log(0) når modellen blir veldig sikker.
        total = 0.0
        for x, y, fasit in data:
            _, z, _ = self.forward(x, y)
            total += max(z, 0) - z * fasit + math.log1p(math.exp(-abs(z)))
        return total / len(data)

    def backprop(self, data):
        dw1 = [[0.0, 0.0] for _ in self.w1]
        db1 = [0.0 for _ in self.b1]
        dw2 = [0.0 for _ in self.w2]
        db2 = 0.0
        for x, y, fasit in data:
            h, _, p = self.forward(x, y)
            feil = (p - fasit) / len(data)
            db2 += feil
            for i, a in enumerate(h):
                dw2[i] += feil * a
                bakover = feil * self.w2[i] * (1 - a * a)
                dw1[i][0] += bakover * x
                dw1[i][1] += bakover * y
                db1[i] += bakover
        return dw1, db1, dw2, db2

    def tren(self, data, steg=2400, læringsrate=0.2):
        for _ in range(steg):
            dw1, db1, dw2, db2 = self.backprop(data)
            # Regn alle gradientene før vi endrer noen av vektene.
            for i in range(len(self.w1)):
                self.w1[i][0] -= læringsrate * dw1[i][0]
                self.w1[i][1] -= læringsrate * dw1[i][1]
                self.b1[i] -= læringsrate * db1[i]
                self.w2[i] -= læringsrate * dw2[i]
            self.b2 -= læringsrate * db2


if __name__ == "__main__":
    navn = sys.argv[1] if len(sys.argv) == 2 else "xor"
    if len(sys.argv) > 2 or navn not in ("xor", "and", "or", "sirkel"):
        raise SystemExit("Kjør python3 nettverk.py [xor|and|or|sirkel]")
    data = datasett(navn)
    modell = Nettverk()
    før = modell.loss(data)
    modell.tren(data, steg=6500 if navn == "sirkel" else 2400)
    treff = sum((modell.forward(x, y)[2] >= 0.5) == t for x, y, t in data)
    print(f"{navn}: loss {før:.4f} → {modell.loss(data):.4f}, {treff}/{len(data)} treningspunkter riktig")
    if navn == "sirkel":
        # Prøv nye punkter her og se hva modellen gjør mellom treningspunktene.
        prøv = [(0, 0), (0.4, 0.1), (0.8, 0.1), (-0.8, -0.8)]
    else:
        prøv = [(x, y) for x, y, _ in data]
    for x, y in prøv:
        print(f"({x:4.1f}, {y:4.1f}) → {modell.forward(x, y)[2]:.3f}")
