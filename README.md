# System rekomendacyjny zadań edukacyjnych (ALS + IRT, PySpark)

Projekt stanowi część badawczą pracy magisterskiej i obejmuje implementację systemu rekomendacyjnego zadań edukacyjnych z wykorzystaniem algorytmu Alternating Least Squares (ALS) w środowisku Apache Spark (PySpark) oraz modelu Item Response Theory (IRT) do adaptacyjnego doboru zadań.

## Cel projektu

Celem projektu jest ocena skuteczności algorytmu ALS w rekomendowaniu zadań edukacyjnych na podstawie historycznych interakcji użytkowników z zadaniami. System bazuje na danych typu implicit feedback i jest oceniany z wykorzystaniem standardowych metryk rankingowych. Wyuczone wektory latentne ALS są następnie łączone z modelem IRT w silniku adaptacyjnym, który dostosowuje trudność zadań do bieżącego poziomu umiejętności użytkownika.

## Opis zbioru danych

Dane dostępne na Kaggle: [danielgruszkowski/data-to-als-recommend-system](https://www.kaggle.com/datasets/danielgruszkowski/data-to-als-recommend-system)

Po pobraniu należy umieścić pliki w katalogu `big_data/`.

### Dane główne
- **`big_data/interactions.parquet`**  
  Zbiór interakcji użytkownik–zadanie wykorzystywany bezpośrednio do uczenia i ewaluacji modelu ALS. Każdy rekord reprezentuje pojedynczą interakcję użytkownika z zadaniem edukacyjnym.

### Dane pomocnicze
- **`big_data/users.parquet`**  
  Zawiera abstrakcyjne parametry opisujące użytkowników, takie jak `guess`, `slip` oraz `learning_rate`, modelujące indywidualne cechy procesu uczenia się.

- **`big_data/tasks.parquet`**  
  Zawiera opisy zadań edukacyjnych, w tym `topic_group`, `difficulty`, główne i poboczne umiejętności oraz wymagania wstępne.

- **`big_data/skills.parquet`**  
  Zbiór definicji umiejętności wykorzystywanych w systemie edukacyjnym.

- **`big_data/irt_params.parquet`**  
  Parametry modelu IRT (dyskryminacja, trudność, zgadywanie) wyznaczone podczas kalibracji.

- **`big_data/theta_final.parquet`**  
  Końcowe oszacowania poziomu umiejętności (θ) użytkowników po kalibracji IRT.

Dane pomocnicze nie są bezpośrednio wykorzystywane w procesie uczenia modelu ALS, lecz stanowią kontekst oraz podstawę dla silnika adaptacyjnego.

## Model rekomendacyjny

W projekcie zastosowano algorytm **Alternating Least Squares (ALS)** w wariancie przeznaczonym do pracy z danymi typu implicit feedback (`implicitPrefs=True`), dostępny w bibliotece `pyspark.ml`.

Każda interakcja użytkownika z zadaniem traktowana jest jako pozytywny sygnał preferencji. Wielokrotne interakcje tego samego użytkownika z tym samym zadaniem są agregowane metodą `log_correct` (log(1 + liczba poprawnych rozwiązań)), co wzmacnia sygnał i ogranicza wpływ bardzo aktywnych użytkowników.

Dla każdego użytkownika generowana jest lista Top-K rekomendowanych zadań, z wykluczeniem zadań obecnych w zbiorze uczącym.

### Parametry modelu (po strojeniu)

| Parametr | Wartość |
|---|---|
| `rank` | 50 |
| `regParam` | 0.05 |
| `maxIter` | 20 |
| `alpha` | 10.0 |

Strojenie hiperparametrów przeprowadzono za pomocą `tune_als.py`.

## Silnik adaptacyjny (ALS + IRT)

Wyuczone wektory latentne ALS są eksportowane do plików `.npz` i łączone z modelem IRT w silniku adaptacyjnym (`src/adaptive_engine.py`). Końcowy wynik rekomendacji jest kombinacją liniową:

```
score = 0.6 × ALS_score + 0.4 × IRT_difficulty_match
```

Poziom umiejętności użytkownika (θ) jest aktualizowany po każdej odpowiedzi z krokiem uczenia 0.05.

## Ewaluacja

Jakość rekomendacji oceniana jest z wykorzystaniem metryk rankingowych przy progach K ∈ {10, 20, 50}:

- HitRate@K
- Precision@K
- Recall@K
- Mean Average Precision (MAP@K)
- Normalized Discounted Cumulative Gain (NDCG@K)

Model ALS porównywany jest z modelami bazowymi: rekomendacjami opartymi na popularności oraz losowymi.

## Struktura projektu

```
.
├── src/
│   ├── als_model.py        # trenowanie modelu ALS
│   ├── irt_model.py        # model IRT
│   ├── adaptive_engine.py  # silnik adaptacyjny ALS + IRT
│   ├── evaluation.py       # metryki rankingowe
│   ├── baselines.py        # modele bazowe
│   ├── data_loader.py      # wczytywanie i podział danych
│   ├── data_prep.py        # przygotowanie danych dla ALS
│   ├── data_config.py      # konfiguracja ścieżek i hiperparametrów
│   ├── session.py          # sesja adaptacyjna użytkownika
│   └── spark_utils.py      # inicjalizacja SparkSession
├── data_generator/
│   ├── generator.py        # generowanie syntetycznych danych
│   └── calibrate_irt.py    # kalibracja parametrów IRT
├── main_als.py             # trening i ewaluacja ALS
├── main_adaptive.py        # demonstracja silnika adaptacyjnego
├── tune_als.py             # strojenie hiperparametrów ALS
├── big_data/               # dane (ignorowane przez git)
└── requirements.txt
```

## Wymagania

```
pyspark>=3.3.0
numpy>=1.24.0
pandas>=2.0.0
pyarrow>=12.0.0
```

```bash
pip install -r requirements.txt
```

## Uruchomienie projektu

**Trening i ewaluacja modelu ALS:**
```bash
python main_als.py
```

**Demonstracja silnika adaptacyjnego:**
```bash
python main_adaptive.py
```

**Strojenie hiperparametrów:**
```bash
python tune_als.py
```
