# Et lite nettverk fra scratch

Vi skal faktisk lære maskinlæring. Da må vi av og til bygge ting selv, helt ned til vektene og gradientene. Her er et lite nettverk du kan trene, pirke i og se hva det gjør.

To innganger, åtte skjulte nevroner og én utgang. JavaScript. Ingen ML-bibliotek, ingen modell-API og ingen API-nøkler. Koden kjører i nettleseren, og alle tallene i visualiseringen kommer fra den samme modellen.

Dette er en liten MA:KI-lab for studenter ved UiO — og alle andre som har lyst til å prøve. Den brukes også på foreningssiden, under **Bygg fra scratch**. Arrangementene våre skal være stedet hvor vi tar slike ting videre sammen. Når arrangementene legges ut, finner du påmeldingen på [Peoply](https://peoply.app).

## Kjør den

Du trenger en helt vanlig lokal webserver, fordi nettleseren bruker JavaScript-moduler. Har du Python:

```sh
python3 -m http.server 8000
```

Åpne `http://localhost:8000` i nettleseren. Dette er bare din lokale adresse, ikke adressen til MA:KI-siden. Det er ingenting å installere for selve laben. Den kan også legges på vanlig statisk hosting; ingen byggekommando trengs.

Har du Node 18 eller nyere, kan du kjøre testene:

```sh
npm test
```

## Hva skal du prøve?

1. **Tren XOR.** Én inngang skal være 1, men ikke begge. En rett linje klarer ikke å skille de fire punktene. Se hvordan det lille nettverket lager flere områder.
2. **Bytt til AND eller OR.** Samme nettverk, mye enklere grenser. Hvorfor det?
3. **Prøv sirkelen.** Modellen får 81 punkter med fasit: innenfor eller utenfor. Den får ikke radiusen eller formelen for en sirkel. Se hvordan området formes av nevronene.
4. **Klikk på kartet.** Krysspunktet blir sendt gjennom modellen. Tallene i hvert nevron er nå de faktiske aktiveringene for akkurat det punktet. Piltastene flytter punktet, og Enter velger sentrum.
5. **Klikk på et nevron.** Under nettverket ser du den konkrete regningen. Prøv å finne et nevron som reagerer mye på ett område og lite på et annet.
6. **Bruk `+1`.** Ett treningssteg om gangen. Det er lettere å få tak på hva som endrer seg når vi sakker ned litt.

Trening kan pauses og nullstilles. Samme seed gir samme startvekter hver gang, så det er mulig å sammenligne forsøkene. Datasettbytte starter treningen på nytt. Å klikke på kartet lager ikke nye treningsdata; det undersøker et punkt med vektene vi allerede har.

## Hva betyr fargene?

- **Koblingene:** cyan er positiv vekt, rød er negativ vekt. Tykkelsen viser størrelsen på vekten relativt til de andre vektene i nettverket. Hold pekeren over en kobling for å se tallet.
- **Nevronene:** cyan er positiv aktivering, rød er negativ aktivering. Lysere farge betyr større absoluttverdi. Innganger og tanh-nevroner går fra −1 til +1. Sigmoid-utgangen går fra 0 til 1 og er derfor alltid cyan eller mørk.
- **Beslutningskartet:** cyan betyr at modellen anslår klasse 1; mørkt betyr klasse 0. Flaten kommer fra modellens prediksjoner over hele kartet. Det er ikke et bilde av fasiten.
- **Små ringer på kartet:** dette er selve treningspunktene. Cyan fyll betyr fasit 1, mørkt fyll betyr fasit 0.

En negativ vekt er ikke nødvendigvis en «hemmende kobling» i alle situasjoner. Når den multipliseres med en negativ aktivering, blir bidraget positivt. Det er fortegn og multiplikasjon, ikke en biologisk hjernemodell.

## Hva skjer i regningen?

Det er **33 parametere** totalt: 16 vekter inn i det skjulte laget, 8 biaser der, 8 vekter ut og én siste bias.

For skjult nevron `i`:

```text
zᵢ = wᵢₓ · x + wᵢᵧ · y + bᵢ
hᵢ = tanh(zᵢ)
```

Vi summerer bidragene fra de åtte nevronene til én logit, og bruker sigmoid for å få et tall mellom 0 og 1:

```text
z = b + Σ vᵢ · hᵢ
p = sigmoid(z)
```

`p` er modellens anslag på klasse 1. Det er ikke en garanti for at modellen er godt kalibrert. På de boolske datasettene mates 0 og 1 inn som −1 og +1; uttrykkene under kartet bruker fortsatt de vanlige 0/1-verdiene.

Loss er **binary cross-entropy**, gjennomsnittet over alle treningspunktene. Modellen straffes for å gi lavt anslag til riktig klasse. Vi regner BCE direkte fra logiten, så vi ikke ender med `log(0)` når modellen blir veldig sikker.

Så kommer backprop:

```text
δ_ut = p − fasit
∂L/∂vᵢ = δ_ut · hᵢ
δᵢ = δ_ut · vᵢ · (1 − hᵢ²)
∂L/∂wᵢₓ = δᵢ · x
∂L/∂wᵢᵧ = δᵢ · y
```

Vi tar gjennomsnittet av gradientene over **hele datasettet**, og oppdaterer med gradient descent:

```text
ny vekt = gammel vekt − læringsrate · gradient
```

Læringsraten er `0.2`. Alle gradientene regnes med de gamle vektene før en eneste vekt oppdateres. Vi trener 30 slike full-batch-steg per animasjonsramme, og oppdaterer kartet omtrent hvert 80. millisekund for å holde nettleseren responsiv.

Grafen viser faktisk loss underveis, med logaritmisk y-akse. Treffprosenten gjelder **treningspunktene**. 100 % her forteller ikke at vi har bevist at modellen fungerer på alle mulige punkter. Kartet lar oss se hva den gjør mellom punktene også.

## Hvor ligger ting?

| Fil | Hva den gjør |
| --- | --- |
| [src/network.js](src/network.js) | Vekter, forward pass, stabil BCE, gradienter og treningssteg. Start her. |
| [src/datasets.js](src/datasets.js) | XOR, AND, OR og de 81 sirkelpunktene. |
| [src/lab-view.js](src/lab-view.js) | Kobler modellen til nevronene, kartet og knappene. |
| [src/lab.css](src/lab.css) | Stilen til selve laben. |
| [src/main.js](src/main.js) | Monterer laben og bytter mellom norsk og engelsk. |
| [tests/network.test.js](tests/network.test.js) | Sjekker determinisme, læring, ekstreme logits og alle gradientene. |

MA:KI-nettsiden har en lokal kopi av de tre kjernemodulene. Den trenger ikke dette repoet ved siden av seg for å bygge. Når vi forbedrer regningen eller visualiseringen, må de kopiene oppdateres samtidig.

## Publisering

Repoet er laget for å kunne deles som ren kildekode. Ingen hemmeligheter, ingen kontooppsett og ingen privat MA:KI-database er med. Demoen består av `index.html`, `src/` og `fonts/`. De filene kan legges rett på statisk hosting, med samme mappestruktur. README og testene trenger ikke være del av nettsiden.

Lenkene til koden peker på [AndersErikstad/maki-fra-scratch](https://github.com/AndersErikstad/maki-fra-scratch). Repoet har ingen forutsetning om at en offentlig demo allerede er publisert. Vi legger heller ikke inn en gjettet adresse til MA:KI-siden; påmeldingslenken går foreløpig til Peoply.

## Hvordan vet vi at backprop er riktig?

Testen flytter hver parameter litt opp og litt ned, regner loss begge ganger, og sammenligner den numeriske gradienten med backprop:

```text
numerisk gradient ≈ [L(w + ε) − L(w − ε)] / (2ε)
```

Dette sjekkes for alle 33 parametere med `ε = 0.00001`. I tillegg skal nettverket lære alle fire datasettene med den faste startseeden. Tester alene er ikke en full bevisføring, men dette fanger opp mange av de feilene som er lette å gjøre når man skriver backprop selv.

## Hva kan vi bygge videre?

Lag et nytt datasett. Bytt tanh med ReLU og regn ut den nye deriverte. Prøv færre nevroner. Gjør læringsraten til en knapp og se hvor fort det kan gå galt. Del opp sirkelen i treningsdata og testdata og sjekk om modellen faktisk lærer noe som fungerer utenfor de punktene den har sett.

Det er her det blir gøy. Vi skal forstå regningen — og så eksperimentere litt.

## Bruk koden

Koden er tilgjengelig under [MIT-lisensen](LICENSE). Fontene Jost og Space Mono er laget av sine respektive opphavere og følger SIL Open Font License; lisensene ligger i [fonts](fonts).
