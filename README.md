# Et lite nevralt nettverk fra scratch

Her er nettverket fra MAKI-laben i python, med to innganger, åtte skjulte nevroner og én utgang. Alt ligger i `nettverk.py`, og du trenger bare python.

```sh
python3 nettverk.py
python3 nettverk.py sirkel
```

Du kan også prøve `and` og `or`. Nettverket bruker tanh, sigmoid og backprop, og skriver ut loss og noen prediksjoner etter trening. Husk at selv om vi får null i loss så kan vi fortsatt bomme på nye punkter!

Prøv å endre antall nevroner, læringsraten eller dataene og se hva som skjer!

[MIT-lisens](LICENSE)
