# Zapytania do Consensus.ai — sekcja C (potok analizy)

Consensus działa najlepiej na **pojedynczych, wąskich pytaniach po angielsku**, a nie na
jednym długim poleceniu. Poniżej zestaw pytań pogrupowany według punktów C.1–C.7 z
`todo_notion.csv`. Przy każdym zapisane, **jaką decyzję ma rozstrzygnąć** — jeśli
odpowiedź niczego nie zmienia, pytanie można pominąć.

Kolejność ma znaczenie: C.2 i C.6 są najważniejsze, bo od nich zależy, czy ranking metod
w pracy będzie obronialny.

---

## Kontekst do wklejenia na początku sesji (opcjonalnie)

> I am evaluating 17 ECG denoising methods (7 static filters, 5 adaptive filters using an
> accelerometer reference, 5 deep neural networks) in two settings: a synthetic one on the
> MIT-BIH Arrhythmia database with noise from the MIT-BIH Noise Stress Test Database,
> where a clean reference exists, and a real one on wearable recordings where it does not.
> I need to choose reference metrics, no-reference signal quality indices, an HRV
> comparison, a QRS detection score and a statistical procedure.

---

## C.1 — Metryki referencyjne

**1.** *What is the standard definition of percentage root mean square difference (PRD) for
ECG denoising, and does the denominator use the clean signal or the denoised signal?*

Rozstrzyga: wzór (14) u Wanga ma w mianowniku sygnał **odszumiony**, co wygląda na
literówkę. Muszę wiedzieć, która konwencja jest standardem, zanim porównam się z ich
liczbami.

**2.** *Should PRD be computed on the raw ECG or after removing the mean, and how much do
the two versions differ?*

Rozstrzyga: PRD i PRD1 (z odjętą składową stałą) różnią się istotnie przy obecności
offsetu. Trzeba zadeklarować którą.

**3.** *Is signal-to-noise ratio improvement (SNRimp) a fair comparison metric when methods
are evaluated at different input SNR levels?*

Rozstrzyga: czy raportować SNR wyjściowe, przyrost, czy oba.

**4.** *What are the reported drawbacks of mean squared error as a loss and as an evaluation
metric for ECG denoising compared to mean absolute error?*

Rozstrzyga: SCED-Net trenowano na MAE, mój domyślny to `improved_mse`. Chcę wiedzieć, czy
zmiana straty jest uzasadniona literaturą, czy tylko naśladownictwem.

**5.** *Do amplitude-normalised and unnormalised ECG denoising metrics lead to different
rankings of methods?*

Rozstrzyga: czy normalizować przed liczeniem metryk (min-max, jak Wang) czy po.

---

## C.2 — Wskaźniki jakości sygnału `najważniejsze`

**6.** *What are the standard definitions of kSQI, pSQI and sSQI for ECG signal quality
assessment, including the exact frequency bands used by pSQI?*

Rozstrzyga: implementację. Potrzebuję pasm liczbowo, nie opisowo.

**7.** *Can spectral ECG signal quality indices such as pSQI be artificially improved by
band-pass filtering without improving diagnostic quality?*

Rozstrzyga: **to jest kluczowe pytanie całej sekcji.** Większość moich filtrów statycznych
to filtry pasmowe, a pSQI mierzy stosunek mocy w pasmach. Jeśli literatura potwierdza tę
podatność, muszę wprowadzić metody kontrolne i walidację panelu.

**8.** *Which ECG signal quality indices are most robust to over-smoothing, and which reward
the removal of genuine signal content?*

Rozstrzyga: dobór panelu tak, żeby nie nagradzał niszczenia sygnału.

**9.** *How well do no-reference ECG signal quality indices correlate with true
signal-to-noise ratio when both can be computed?*

Rozstrzyga: czy walidacja panelu przez korelację z SNR na danych syntetycznych jest
uznaną metodą, czy trzeba czegoś mocniejszego.

**10.** *Have combinations of multiple ECG signal quality indices been shown to outperform
single indices, and how are they usually combined?*

Rozstrzyga: czy raportować wskaźniki osobno, czy agregować.

**11.** *What thresholds of kSQI and pSQI are used to classify an ECG segment as acceptable
for analysis?*

Rozstrzyga: czy w ogóle stosować progi, czy traktować wskaźniki jako ciągłe.

---

## C.3 — HRV

**12.** *What is the minimum recording length required for reliable short-term heart rate
variability analysis in the time domain and in the frequency domain?*

Rozstrzyga: czy 5 minut wystarczy, czy potrzeba więcej; wpływa na protokół pomiarowy.

**13.** *How does ECG denoising affect heart rate variability indices, and can denoising
introduce bias in RR interval estimation?*

Rozstrzyga: czy HRV nadaje się na metrykę jakości filtracji, czy raczej na kontrolę
bezpieczeństwa (czy filtr niczego nie zepsuł).

**14.** *What proportion of ectopic or artefact-corrupted RR intervals invalidates a heart
rate variability analysis, and what correction methods are recommended?*

Rozstrzyga: implementację odsetka interwałów artefaktowych z punktu C.3.

**15.** *Which heart rate variability indices are most sensitive to noise in the ECG signal?*

Rozstrzyga: który wskaźnik wybrać jako czuły detektor uszkodzenia sygnału.

---

## C.4 — Detekcja QRS jako metryka

**16.** *What matching tolerance is used to score QRS detection against reference
annotations, and what standard defines it?*

Rozstrzyga: przyjąłem 150 ms za ANSI/AAMI EC57 — chcę potwierdzenia i numeru normy do
cytowania.

**17.** *Is QRS detection performance a valid surrogate measure of ECG denoising quality?*

Rozstrzyga: czy F1 detekcji może stać w tabeli obok SNR, czy tylko jako kontrola.

**18.** *How much does QRS detection accuracy degrade as the signal-to-noise ratio falls,
and at what SNR do common detectors fail?*

Rozstrzyga: przy jakim SNR moja siatka (−9…11 dB) wchodzi w obszar, gdzie detekcja
przestaje być wiarygodna — a razem z nią segmentacja na cykle w SCED-Net.

---

## C.5 — Walidacja sanity

**19.** *What sanity checks are recommended to verify that an ECG denoising evaluation
pipeline computes its metrics correctly?*

Rozstrzyga: czy mój pomysł (biały szum plus filtr dolnoprzepustowy o znanym analitycznym
przyroście SNR) jest uznaną praktyką, czy jest coś lepszego.

**20.** *Have published ECG denoising comparisons included a no-filtering baseline and a
deliberately destructive filter as controls?*

Rozstrzyga: czy metody kontrolne w tabeli wyników są przyjęte, czy będę pierwszy — i czy
trzeba to szerzej uzasadnić.

---

## C.6 — Statystyka `najważniejsze`

**21.** *When comparing signal processing methods on overlapping windows from the same
patients, what is the correct unit of statistical analysis?*

Rozstrzyga: potwierdzenie, że agregacja per pacjent przed testem jest wymagana, a nie
tylko ostrożna.

**22.** *Which statistical test is appropriate for comparing multiple ECG denoising methods
on the same set of recordings?*

Rozstrzyga: Wilcoxon sparowany, Friedman z testem post hoc, czy coś innego przy 17
metodach.

**23.** *What multiple comparison correction is recommended when comparing many signal
processing methods pairwise, and is Holm preferred over Bonferroni or
Benjamini-Hochberg?*

Rozstrzyga: implementację korekty z punktu C.6.

**24.** *Should effect sizes and confidence intervals be reported alongside p-values in
biomedical signal processing comparisons, and which effect size measure is used for
paired non-parametric data?*

Rozstrzyga: czy dokładać rozmiar efektu; przy 5 pacjentach testowych p-wartość sama
niewiele powie.

**25.** *How should confidence intervals be computed for a metric expressed in decibels?*

Rozstrzyga: czy liczyć przedział w dB, czy w dziedzinie liniowej i przeliczać — to nie
jest to samo, bo logarytm jest nieliniowy.

---

## C.7 — Przypadki brzegowe

**26.** *How should signal-to-noise ratio and PRD be defined when the reference signal is
constant or the residual is zero?*

Rozstrzyga: zachowanie metryk na przypadkach brzegowych, których wymaga C.7.

---

## Dodatkowo, poza CSV — warte sprawdzenia

**27.** *Do published deep learning ECG denoising studies separate training and test data by
patient, and what is the reported effect of patient leakage on results?*

Rozstrzyga: chcę liczbę mówiącą, o ile przeciek zawyża wynik — do rozdziału o dyskusji,
gdzie moje wyniki wypadną gorzej od publikowanych i muszę to uzasadnić.

**28.** *Are motion artefacts in wearable ECG dominated by electrode-skin impedance changes
or by electromyographic activity, and does this depend on electrode placement?*

Rozstrzyga: skład mieszanki szumu (`em` kontra `em`+`ma`) w zależności od tego, że
elektrody będą na klatce, nie na przedramieniu.

**29.** *Can an accelerometer serve as an adequate reference for adaptive cancellation of
electromyographic noise, or only for electrode motion artefacts?*

Rozstrzyga: przewidywane ograniczenie filtrów adaptacyjnych przy skurczu izometrycznym —
chcę wiedzieć, czy ktoś to już zmierzył.

**30.** *Does using more than one accelerometer improve adaptive motion artefact cancellation
in ECG, and how are multiple reference channels usually combined?*

Rozstrzyga: czy dwa akcelerometry to znany zabieg, czy element własny pracy.
