# Plan pracy — stan na 29.08.2026

Zaawansowane metody przetwarzania i oceny sygnału z urządzeń EKG typu Holter.
Aktualizacja po ustaleniach dotyczących bazy BUT QDB, toru pomiarowego neurobita
i rezygnacji z segmentacji opartej na tabelach indeksów.

---

## A. Zrobione

### A.1 Audyt i naprawa istniejącego kodu

- 16 błędów w `src/filters/`: przesłonięcie `scipy.signal`, podwójna normalizacja `Wn`,
  zły rozmiar wektora wag w BLMS, brak `return` w hybrydzie GALL+Kalman, macierz błędu
  `(N, M+2)` zamiast wektora, `ConvTranspose1d` dający L+1 próbek.
- Filtry statyczne przeniesione na SOS i filtrację zerofazową.
- Ustalone: DeepCEDNet wymaga 992 próbek, nie 1024 (skok STFT 32, 32 ramki).

### A.2 Warstwa reprezentacji i trening

- `src/train/signal_selection.py` — `SignalRepresentation` z czterema implementacjami,
  leniwe rozwiązywanie modeli przez `importlib`, brak zależności od PyTorcha.
- `src/train/training.py` — `Trainer` z AMP, akumulacją gradientu, metrykami na GPU.
- `src/train/cli.py` — punkt wejścia.

### A.3 Przygotowanie danych

| Moduł | Zawartość |
|---|---|
| `normalization.py` | resampling przez `Fraction.limit_denominator`, ścieżka polifazowa i FFT, skalowanie µV→mV |
| `qrs_detection.py` | neurokit2, dobór detektora zmierzony na MIT-BIH (F1 = 98.18 %), dopasowanie 1:1, tolerancja 150 ms |
| `noise_mixing.py` | dwie konwencje SNR (`power_ratio` domyślna, `nst` zachowana), parametry Wang et al. 2023 |
| `splitting.py` | `holdout_split` (pacjenci), `group_kfold`, `verify_folds` |
| `windows.py` | generatory okien: `sliding_windows`, `beat_windows` |

**156 testów przechodzi.**

### A.4 Dokumentacja

- Oświadczenie o wykorzystaniu AI (`.md` i `.tex`), dziennik prac (w `.gitignore`).
- Spis treści zatwierdzony przez promotora, szablon `aghdpl` w Overleafie.

### A.5 Ustalenia metodologiczne

- Podział wyłącznie po pacjentach; rekordy 201 i 202 to jedna osoba.
- Ocena agregowana na poziomie rekordu, nie okna — okna w obrębie pacjenta nie są
  niezależne, więc test statystyczny na oknach zawyża istotność.
- Konwencja SNR zgodna z literaturą uczenia głębokiego, nie z narzędziem `nst`.
  Próba odtworzenia rekordów `118e*` nie powiodła się (stały czynnik ~24 w mocy);
  udokumentowane w `scripts/diagnose_noise_power.py`, do rozdziału o ograniczeniach.
- Podziału 95/5 z Wang et al. nie kopiujemy — miesza pacjentów między zbiorami.

---

## B. Ścieżka krytyczna

Kolejność ma znaczenie: każdy punkt blokuje następne.

### ~~B.1 Commit zaległych plików~~ `zrobione`

### ~~B.2 Uruchomić `tests/smoke_test.py`~~ `zrobione`

Cały zbiór przechodzi łącznie z `test_suite.py`, czyli **wszystkie pięć architektur
wykonało przebieg w przód**. Do tego momentu wymiary były potwierdzone wyłącznie
analitycznie. Warstwa modeli jest zweryfikowana empirycznie.

Stan: 199 testów w zbiorze podstawowym, 3 sieciowe, 7 sprawdzeń w `test_suite.py`.

### ~~B.3 Domknąć `requirements.txt`~~ `zrobione`

`neurokit2==0.2.13`.

### ~~B.4 Spiąć pipeline danych~~ `zrobione` (A.6, A.7 z todo_notion)

Eksport `.npz` czytany przez `train/cli.py` bez pickle, z wymaganym `fs` i podziałem
po pacjentach. Odczyt wydzielony do `src/train/dataset_io.py`, wolnego od PyTorcha, żeby
kontrakt warstwy danych dało się testować wszędzie, gdzie działa numpy.

Niezmienniki w `tests/test_data.py`: brak NaN/Inf, brak okien stałych, zakres amplitudy,
zgodność SNR przeliczonego z przebiegów, spójność `r_peaks` z offsetami, rozdzielność
pacjentów. 247 testów w zbiorze.

### B.5 Panel SQI `nie istnieje w ogóle`

kSQI, pSQI, sSQI. Bez niego nie ma jak ocenić środowiska rzeczywistego, w którym
spotykają się wszystkie trzy rodziny filtrów. Do `src/evaluation/sqi.py`.

### B.6 Walidacja panelu SQI

Dwie niezależne kontrole:

- korelacja rangowa SQI z prawdziwym SNR na MIT-BIH, gdzie znane jest jedno i drugie;
- zgodność z konsensusem trzech ekspertów w BUT QDB.

**Powód:** pSQI to stosunek mocy w paśmie QRS do pasma szerszego, więc filtr pasmowy
5–15 Hz osiąga wynik idealny niszcząc sygnał. Większość filtrów statycznych w tej pracy
to filtry pasmowe, więc bez walidacji ranking byłby bliski tautologii.

### B.7 Metody kontrolne w porównaniu

Identyczność (brak filtracji) jako dolna granica oraz celowo niszczący filtr pasmowy
5–15 Hz. Jeśli filtr niszczący wygra na panelu SQI, panel jest zepsuty i jest na to dowód
w tabeli.

### B.8 Agregacja i statystyka

Metryka uśredniana w obrębie pacjenta, dopiero potem test sparowany. Wilcoxon na
kilkudziesięciu pacjentach, nie na tysiącach okien.

---

## C. Wyszło dodatkowo

### C.1 Baza BUT QDB `duża wartość`

18 nagrań dobowych, 15 osób, EKG 1000 Hz **z zsynchronizowanym akcelerometrem 3-osiowym
100 Hz**, warunki życia codziennego, format WFDB, CC BY 4.0.

Znaczenie: filtry adaptacyjne przestają być oceniane na kilku własnych nagraniach.
Dodatkowo anotacje jakości od trzech ekspertów pozwalają zwalidować panel SQI względem
oceny człowieka.

Ograniczenia: brak czystego odniesienia (metryki pełnoreferencyjne niepoliczalne), brak
adnotacji zespołów QRS, anotowana tylko część materiału, 4.2 GB.

Resampling: 1000→360 Hz to 9/25, 100→360 Hz to 18/5 — obie ścieżki polifazowe, kod bez zmian.

### C.2 Dwa warianty MIT-BIH

Neurobit stosuje prefiltr sprzętowy `HP 0.35 Hz / LP 100 Hz / N 50 Hz`. MIT-BIH ma pasmo
0.1–100 Hz bez notcha i zakłócenie sieciowe **60 Hz** (nagrania amerykańskie).

- wariant surowy — do porównań z literaturą;
- wariant z emulacją toru neurobita (z notchem 60 Hz) — do twierdzeń o przeniesieniu
  modeli na dane własne.

Bez tego nie da się odróżnić „model nie radzi sobie z innym pacjentem" od „model dostał
inny tor analogowy".

### C.3 Częstotliwość sieci jako parametr

50 Hz w Polsce, 60 Hz w MIT-BIH. Nie może być stałą w filtrach statycznych.

### C.4 Standaryzacja kanału referencyjnego

Krok LMS spełnia $0 < \mu < 2/(M P_x)$, więc zależy od mocy referencji. Akcelerometr
podany w zliczeniach ADC zamiast w $g$ zmienia $P_x$ o sześć rzędów i $\mu$ przestaje
działać. Standaryzować referencję do wariancji jednostkowej przed filtrem adaptacyjnym
i zapisać to jako świadomą decyzję.

### C.5 Protokół pomiarowy

Ustalenia po analizie `recordH`:

- **elektrody na klatce w montażu odtwarzającym MLII** — ujemna pod prawym obojczykiem,
  dodatnia w dolnej lewej części klatki. Dziedzina morfologiczna zgadza się wtedy
  z kanałem 0 MIT-BIH bez żadnego dostrajania;
- **akcelerometr na klatce, przy elektrodzie**, nie na nadgarstku — referencja ma być
  skorelowana z ruchem granicy skóra–elektroda;
- nagrania **co najmniej 6 minut** (analiza HRV krótkoterminowa wymaga 5, potrzebny zapas
  na przycięcie); `recordH` ma 4.73 min;
- osobny scenariusz ze **skurczem izometrycznym** — pełne EMG przy zerowym przyspieszeniu,
  czyli sytuacja, w której referencja akcelerometryczna z definicji nie pomaga.

### C.6 Synchronizacja ESP32 z neurobitem

Eksport neurobita ma `NumSig=1` — akcelerometru w nim nie ma. Dwa niezależne zegary,
brak wspólnej podstawy czasu. Potrzebne zdarzenie widoczne w obu torach (stuknięcie
w czujnik na starcie). Przy nagraniach kilkuminutowych dryf jest pomijalny.

### ~~C.7 Decyzja o częstotliwości próbkowania~~ `rozstrzygnięte`

**360 Hz wszędzie.** Neurobit zostanie ustawiony na tę samą wartość co MIT-BIH, więc
resampling sygnału EKG odpada, a porównanie jest bezpośrednie. Konsekwencje:

- akcelerometry też muszą pracować przy 360 Hz, żeby referencja zgadzała się z EKG
  próbka w próbkę;
- filtr antyaliasingowy toru analogowego akcelerometru dobrać wyraźnie poniżej
  częstotliwości Nyquista 180 Hz; treść artefaktu ruchowego mieści się poniżej 50 Hz,
  więc pasmo rzędu 50-100 Hz jest właściwe;
- prefiltr neurobita `LP 100 Hz` przy 360 Hz ma większy zapas do Nyquista niż miał
  przy 250 Hz.

### C.8 Nagrania z przedramienia jako przypadek skrajny

Cztery istniejące rekordy zbierane z przedramienia. Zmierzone: stosunek energii pasma
QRS do pasma EMG spada z 2.64 w spoczynku do **0.76 przy ruchu** — EMG przewyższa sygnał.
Materiał na podrozdział o granicach metody, nie do wyrzucenia.

### C.9 FSSTH

`TimeFrequencySignal` używa zwykłego STFT jako zastępnika synchrosqueezingu. Albo
implementacja FSSTH, albo jawna deklaracja zastępnika w metodologii. **Decyzja otwarta.**

---

## D. Redakcja

- Uzupełnić `\supervisor` i `\acknowledgements` w `main.tex`.
- Zweryfikować każdą pozycję `bibliografia.bib` względem oryginału; uzupełnić
  `terada2025wavelet`.
- Oświadczenie o AI — na końcu, po zamknięciu części technicznej.

---

## E. Ryzyka

| Ryzyko | Skutek | Zabezpieczenie |
|---|---|---|
| ~~Modele nigdy nie uruchomione~~ | — | zamknięte, B.2 |
| Panel SQI nagradza niszczenie sygnału | fałszywy ranking w pracy | B.6, B.7 |
| Przesunięcie dziedziny MIT-BIH → neurobit | spadek jakości na danych własnych | C.2, C.5 |
| Referencja ACC nie tłumaczy EMG izometrycznego | filtry adaptacyjne słabe w części scenariuszy | C.5, opisać zamiast ukrywać |
| Brak synchronizacji ACC z EKG | filtry adaptacyjne bez danych | C.6 |

---

## F. Sugerowana kolejność

1. ~~B.1, B.3, B.2~~ — zrobione.
2. **B.4** — spięcie pipeline'u: eksport `.npz` → `ECGDenoisingDataset` → `Trainer`.
4. C.5 — protokół pomiarowy; zaplanować i **nagrać** wcześnie, bo to jedyny element
   z opóźnieniem fizycznym.
5. C.6 — synchronizacja, równolegle z nagraniami.
6. B.5, B.6, B.7 — panel SQI i jego walidacja.
7. C.1 — BUT QDB; można zacząć pobieranie w tle w dowolnym momencie.
8. C.2, C.3 — warianty MIT-BIH.
9. B.8 — agregacja i statystyka.
10. C.7, C.9 — decyzje otwarte, do rozstrzygnięcia przed opisem metodologii.
11. D — redakcja.
