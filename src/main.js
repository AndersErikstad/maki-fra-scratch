import { mountLabUI } from './lab-view.js';

const copy = {
  nb: {
    header: 'VI BYGGER FRA SCRATCH.',
    eyebrow: 'LITT FORAN PENSUM. HELT NED I REGNINGEN.',
    title: 'Åtte nevroner.\nHva gjør de, egentlig?',
    lead: 'Vi skal faktisk lære maskinlæring. Her er et lite nettverk vi kan trene, pirke i og forstå. Alle tallene du ser kommer fra modellen som kjører her.',
    afterTitle: 'Prøv dette.',
    tryOne: 'Tren XOR. Klikk på kartet og se hvilke nevroner som reagerer på de ulike områdene.',
    tryTwo: 'Bytt til AND. Hvorfor holder det nå med én grense mellom punktene?',
    tryThree: 'Prøv sirkelen. Nettverket får ikke formelen for en sirkel. Hvordan lager åtte tanh-nevroner likevel et område som nesten ligner?',
    afterCopy: 'Så kan vi åpne koden og bygge videre. Det er hele poenget.',
    source: 'Koden og forklaringen ligger på GitHub ↗',
    footer: 'MA:KI · Maskinlæring og kunstig intelligens ved UiO.',
    peoply: 'Se hva som skjer på IFI ↗',
  },
  en: {
    header: 'WE BUILD FROM SCRATCH.',
    eyebrow: 'A LITTLE AHEAD OF THE SYLLABUS. RIGHT DOWN TO THE MATHS.',
    title: 'Eight neurons.\nWhat are they actually doing?',
    lead: 'We want to actually learn machine learning. Here is a little network we can train, poke around in, and understand. Every number you see comes from the model running here.',
    afterTitle: 'Try this.',
    tryOne: 'Train XOR. Click the map and see which neurons respond to different regions.',
    tryTwo: 'Switch to AND. Why is one boundary between the points enough now?',
    tryThree: 'Try the circle. The network never gets a formula for a circle. How do eight tanh neurons still produce something that almost looks like one?',
    afterCopy: 'Then we can open the code and build on it. That is the whole point.',
    source: 'The code and explanation are on GitHub ↗',
    footer: 'MA:KI · Machine learning and artificial intelligence at UiO.',
    peoply: 'See what is happening at IFI ↗',
  },
};

let language = new URLSearchParams(location.search).get('lang') === 'en' ? 'en' : 'nb';
let cleanup;
const languageButton = document.querySelector('[data-language]');

function render() {
  cleanup?.();
  document.documentElement.lang = language;
  document.title = language === 'nb' ? 'MA:KI · Et nettverk fra scratch' : 'MA:KI · A network from scratch';
  document.querySelectorAll('[data-page]').forEach(element => {
    element.textContent = copy[language][element.dataset.page];
  });
  languageButton.textContent = language === 'nb' ? 'EN' : 'NO';
  languageButton.setAttribute('aria-label', language === 'nb' ? 'Switch to English' : 'Bytt til norsk');
  cleanup = mountLabUI(document.querySelector('#lab'), language, { eventsURL: 'https://peoply.app' });
}

languageButton.addEventListener('click', () => {
  language = language === 'nb' ? 'en' : 'nb';
  const url = new URL(location.href);
  if (language === 'en') url.searchParams.set('lang', 'en');
  else url.searchParams.delete('lang');
  history.replaceState(null, '', url);
  render();
});
render();
