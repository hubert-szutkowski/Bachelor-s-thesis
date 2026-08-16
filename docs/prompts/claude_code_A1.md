# Prompt dla Claude Code — domknięcie etapu A.1

Skopiuj wszystko poniżej linii i wklej do Claude Code w VS Code.

---

## Kontekst

Repozytorium pracy inżynierskiej: porównanie metod usuwania artefaktów ruchowych i pływania
linii izoelektrycznej z sygnału EKG. Pracujesz na branchu `data_entrance`.

Struktura:

```
data/scripts/physionet.py        pobieranie z PhysioNet (istnieje, do poprawy)
src/filters/                     metody odszumiania: statyczne, adaptacyjne, sieci
src/train/                       warstwa reprezentacji sygnału + pętla treningowa
tests/                           smoke_test.py, test_signal_selection.py
tests/data_tests/download_test.py   (istnieje, do przepisania)
requirements.txt                 wersje przypięte, pytest 9.1.1
```

Docelowy potok: pobranie MIT-BIH Arrhythmia i NSTDB → wczytanie → detekcja QRS →
kalibrowane mieszanie szumu → segmentacja → eksport `.npz` dla `src/train/cli.py`.
Ten prompt obejmuje wyłącznie **pobieranie i wczytywanie**.

## Zasady, których masz przestrzegać

**Styl kodu.** Nazewnictwo i docstringi po angielsku. Nie dopisuj komentarzy liniowych —
docstringi wystarczą. Type hints obowiązkowo. Zachowaj konwencję istniejącego kodu.

**Filozofia testów.** Testuj to, co psuje się po cichu, nie to, co psuje się głośno.
Nie testuj bibliotek zewnętrznych — `wfdb` działa, to nie jest przedmiot testów.
Testuj **swoje** parsowanie, walidację i niezmienniki danych. Żadnych testów
napisanych tylko po to, żeby podnieść pokrycie.

**Nigdy nie łap wyjątków w testach.** Traceback jest informacją diagnostyczną.
Zero `try/except` w kodzie testowym.

**Testy asertują, nie zwracają.** Funkcja testowa zwracająca `bool` przechodzi
w pytest niezależnie od wartości — to jest bug, który obecnie mamy w repo.

## Zadanie 1 — naprawa `data/scripts/physionet.py`

**1.1 Ścieżki bezwzględne.** Obecnie `os.path.join('data', 'files', ...)` zależy od
katalogu roboczego. Kod będzie uruchamiany zdalnie, ze schedulera, z nieznanego cwd.
Zakotwicz wszystko w katalogu repozytorium:

```python
ROOT = Path(__file__).resolve().parents[2]
```

Przejdź w całości na `pathlib.Path`. Dodaj opcjonalny parametr `root: Path | None = None`
do funkcji publicznych, domyślnie `ROOT`, żeby testy mogły podać katalog tymczasowy.

**1.2 Funkcje zwracają ścieżki.** Obecnie zwracają `None`, więc wywołujący nie wie,
co się pobrało, a test nie ma czego asertować. Zwracaj `list[Path]` z katalogami
albo ścieżkami bazowymi rekordów.

**1.3 Jeden katalog na bazę, nie na rekord.** `wfdb.dl_database` przyjmuje listę
rekordów i zapisuje je do wspólnego katalogu. Obecny kod tworzy osobny katalog na
każdy rekord, co przy 48 nagraniach MIT-BIH daje 48 katalogów po jednym rekordzie
i komplikuje późniejsze wczytywanie. Docelowy układ:

```
data/files/mitdb/       100.dat 100.hea 100.atr 101.dat ...
data/files/nstdb/       bw.dat bw.hea em.dat ... 118e06.dat ...
```

**1.4 Pomijanie tego, co już jest.** Przed pobraniem sprawdź, czy komplet plików
rekordu istnieje. Pobieranie MIT-BIH trwa i nie ma powodu powtarzać go przy każdym
uruchomieniu. Dodaj parametr `force: bool = False`.

**1.5 Domyślne wartości.** `records_count: int = 5` jako domyślna wartość w funkcji
produkcyjnej to pułapka — łatwo pobrać 5 rekordów zamiast 48 i nie zauważyć.
Domyślnie pobieraj **całą** bazę, a ograniczenie niech będzie jawne (`records=None`
oznacza wszystko).

## Zadanie 2 — wczytywanie (to jest właściwa treść A.1)

Utwórz `data/scripts/loader.py`.

**2.1 Struktura wyniku.** Dataclass `EcgRecord` z polami:

```
record_id   str
database    str
signal      np.ndarray, kształt (n_samples, n_channels), float64, jednostki mV
fs          float
channels    list[str]           nazwy odprowadzeń, np. ['MLII', 'V1']
units       list[str]
r_peaks     np.ndarray | None   indeksy próbek pobudzeń
symbols     np.ndarray | None   symbole adnotacji ('N', 'V', 'A', ...)
path        Path
```

**2.2 Funkcja `load_record(record_id, database='mitdb', root=None, with_annotations=True) -> EcgRecord`.**
Używa `wfdb.rdrecord` i `wfdb.rdann`. Sygnał zwracany w wartościach fizycznych (mV),
nie w jednostkach ADC.

**2.3 Walidacja przy wczytywaniu.** To jest sedno zadania — uszkodzone albo ucięte
pobranie ma się ujawnić natychmiast, a nie po ośmiu godzinach treningu. Funkcja
`validate_record(record: EcgRecord, expected_fs: float | None = None) -> None`
rzuca `ValueError` z czytelnym komunikatem, gdy:

- liczba próbek nie zgadza się z deklaracją w nagłówku
- w sygnale są `NaN` lub `inf`
- kanał jest stały (odchylenie standardowe zerowe) — objaw urwanego pliku
- `expected_fs` podane i niezgodne z rzeczywistym
- indeksy adnotacji wychodzą poza zakres sygnału
- amplituda sygnału jest poza rozsądnym zakresem fizjologicznym (sanity check, np. powyżej 50 mV)

**2.4 Rekordy szumu wymagają uwagi.** W `bw.hea`, `ma.hea` i `em.hea` wzmocnienie ADC
jest zapisane jako `0`, czyli **nieokreślone** — WFDB przyjmuje wtedy domyślne
200 ADU/mV. Rekordy pochodne z MIT-BIH mają jawne `200(1024)`. Udokumentuj to
w docstringu `load_noise` i dodaj asercję na zgodność jednostek, żeby przy późniejszym
mieszaniu szumu nie wyszło przeskalowanie o stały czynnik.

**2.5 Funkcja `load_noise(noise_type, root=None) -> EcgRecord`** dla `'bw' | 'ma' | 'em'`,
bez adnotacji (te rekordy ich nie mają).

**2.6 Znane parametry do asercji.** MIT-BIH Arrhythmia: 48 rekordów, 360 Hz, 2 kanały,
650 000 próbek na rekord, wzmocnienie 200 ADU/mV. NSTDB: rekordy szumu 360 Hz,
2 kanały, 650 000 próbek. Nagłówek `118e06.hea` zawiera komentarz
`# Created by 'nst' from records 118 and em (SNR = 6 dB)`.

## Zadanie 3 — przepisanie testów

**3.1 Usuń `tests/data_tests/download_test.py`** i zastąp go dwoma plikami:

`tests/data_tests/test_download.py` — testy wymagające sieci, oznaczone markerem.
`tests/data_tests/test_loader.py` — testy wczytywania i walidacji.

**3.2 Konfiguracja markerów.** Dodaj `pytest.ini` albo sekcję `[tool.pytest.ini_options]`
w `pyproject.toml`:

```ini
[pytest]
testpaths = tests
markers =
    network: requires internet access to PhysioNet
    slow: takes more than a few seconds
```

Domyślne uruchomienie `pytest` ma pomijać `network` i `slow`. Dodaj do `conftest.py`
opcję `--run-network` i `--run-slow`, które je włączają. Powód: na zdalnej maszynie
i w CI test zależny od PhysioNet będzie losowo padał, a to uczy ignorowania czerwonych
wyników.

**3.3 Testy sieciowe** (marker `network`), minimalny zakres:

- `wfdb.get_record_list('mitdb')` zwraca dokładnie 48 rekordów
- pobranie **jednego** rekordu do `tmp_path` tworzy komplet `.dat` + `.hea` + `.atr`
- ponowne wywołanie z `force=False` nie pobiera niczego drugi raz

Nie pobieraj w testach całej bazy ani kompletu szumów. To jest 60 MB i kilka minut.

**3.4 Testy loadera** — bez sieci, na małym rekordzie zapisanym w `tests/data_tests/fixtures/`.
Wygeneruj go raz przez `wfdb.wrsamp` i zacommituj (kilkadziesiąt kB).

- `load_record` zwraca sygnał o kształcie `(n, 2)`, dtype `float64`
- `fs` równe 360
- adnotacje mieszczą się w zakresie `[0, n)`
- `validate_record` **rzuca** `ValueError` dla sygnału z `NaN`
- `validate_record` **rzuca** `ValueError` dla kanału stałego
- `validate_record` **rzuca** `ValueError` przy niezgodnym `expected_fs`

Ostatnie trzy sprawdzaj przez `pytest.raises(ValueError, match=...)`. Test, który
sprawdza wyłącznie ścieżkę szczęśliwą, jest wart połowę tego, na co wygląda.

**3.5 Żadnego kasowania danych.** Obecny skrypt na końcu robi `shutil.rmtree('data/files')`
bezwarunkowo, kasując też dane pobrane wcześniej celowo. Testy mają pracować wyłącznie
w `tmp_path`.

## Kryteria akceptacji

1. `pytest` bez flag przechodzi i **nie dotyka sieci**
2. `pytest --run-network` przechodzi przy dostępie do internetu
3. Żadna funkcja testowa nie zawiera `try`/`except` ani `return` wartości
4. `python -c "from data.scripts.loader import load_record"` działa z dowolnego cwd
5. `load_record('100')` zwraca `EcgRecord` z `fs == 360`, `signal.shape[1] == 2`,
   niepustymi `r_peaks`
6. `validate_record` rzuca `ValueError` na każdym z trzech przypadków uszkodzenia
7. Nic poza `tmp_path` nie zostaje utworzone ani skasowane przez testy

## Na koniec

Uruchom `pytest -v` oraz `pytest -v --run-network` i pokaż mi wyniki. Wypisz krótko,
co zmieniłeś w `physionet.py` i dlaczego. Nie commituj — chcę najpierw przejrzeć diff.
