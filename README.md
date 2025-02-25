# dataScienceCourse

# [Introduction to Machine Learning and Data Science]

## Repository Link
https://github.com/cloudbasti/dataScienceCourse


## Description

In dem Projekt sollen Umsatzdaten einer Bäckerei vorhergesagt werden, aufgrund von vergangenen Umsatzdaten verschiedener Produkte unter Hinzuziehung von Wetterdaten, Events wie Kieler Woche etc. Deshalb wird einmal eine lineare Regression genutzt und ein neuronales Netz um die Aufgabe zu lösen. 


Aufbau:

In den Ordnern Neural Network ist natürlich das Neural Network und in der Regression ist die Multiple Linear Regression. Hier sind die Dateien "neural_simple.py" und "MultipleLinearRegression" jeweils die Python Dateien, mit denen optimiert wirdbzw. welche geändert werden. "Final_Network" und "Final_Regression" sind dann jeweils für die Final Submission, die dann auf dem vollen Trainingssatz trainiert. Hätte man auch mit einer Konfiguration machen können, wäre auch besser. Desweiteren ist die Preparation für den testdatenSatz und den trainingsdatensatz auch verschieden. Hier wäre es sinnvoller diese auch nochmal zu vereinen, allerdings war es alleine auch ein Zeitproblem das noch alles zu refaktorieren (dont touch code that works because it could break :)). Es ist über das Projekt so "historisch" gewachsen" und die Zeit für Refactoring war am Ende nicht mehr da. 

Data: 

Beide Versionen (Netzwerk und Regression) lesen jeweils Data Dateien ein, für die Features und vorab Daten Bearbeitung. Diese sind in dem Ordner "data" enthalten. Hier gibt es 4 Funktionen in data/data_prep.py. 

Daten einlesen und Features: 

def merge_datasets(), 
def handle_missing_values(df):
def prepare_features(df):
def add_time_series_features(df):

Feiertage und Schulferien

Die Dateien "data/create_bank_holidays.py" und "data/create_school_holidays.py" erstellen die zugehörigen csv Files, die dann von def merge_datasets() auch eingelesen werden. Hier kann man auch Werte für Feiertage und Schulferien dann ergänzen, oder nach dem gleichen Schema weitere Daten hinzufügen (wie andere Events oder so). 

Training und Validation Split

Dieser Split wird von der Datei data/train_split.py gemacht. 

Final Test Data und Wetter Imputation Flow: 

# Der Verarbeitungsablauf

Das Projekt besteht aus drei Haupt-Python-Skripten, die in einer bestimmten Reihenfolge ausgeführt werden:

### 1. Anfängliche Wetterdaten-Imputation (`Wetter_Imputation.txt`)
**Zweck**: Fehlende Werte im ursprünglichen Wetterdatensatz auffüllen
- **Eingabe**: `data/wetter.csv`
- **Funktionen**: 
  - `print_missing_analysis()`: Analysiert fehlende Daten
  - `impute_data()`: Füllt fehlende Wettercodes und Bewölkungswerte unter Verwendung monatlicher Verteilungen
- **Ausgabe**: `data/wetter_imputed.csv`

### 2. Daten-Zusammenführung und Feature-Engineering (`prepareTestData.txt`)
**Zweck**: Alle Datensätze kombinieren und Merkmale für die Modellierung erstellen
- **Eingaben**: 
  - `data/test.csv` (Haupt-Testdatensatz)
  - `data/wetter_imputed.csv` (Wetterdaten aus Schritt 1)
  - `data/kiwo.csv` (Kieler Woche Ereignisse)
  - `data/school_holidays.csv` (Schulferien)
  - `data/bank_holidays.csv` (Feiertage)
- **Funktionen**:
  - `merge_test_datasets()`: Kombiniert alle Datensätze anhand des Datums
  - `prepare_features()`: Erstellt zeitbasierte, Wetterkategorie- und Sondertag-Merkmale
  - `initialize_umsatz_column()`: Fügt eine Zielspalte hinzu, die mit Nullen initialisiert wird
- **Ausgabe**: `data/prepared_test_data.csv`

### 3. Finale Wetterdaten-Imputation (`Final_wetter_imputation`)
**Zweck**: Behandlung verbleibender fehlender Wetterwerte nach der Zusammenführung
- **Eingabe**: `data/prepared_test_data.csv`
- **Funktionen**:
  - `analyze_weather_code_distribution()`: Hilft bei der Validierung der Datenverteilung
  - `impute_weather_data()`: Umfassendere Imputation für alle Wetterspalten:
    - Temperatur (unter Verwendung wöchentlicher Durchschnittswerte)
    - Windgeschwindigkeit (unter Verwendung des Medians)
    - Bewölkung (unter Verwendung des Medians)
    - Wettercodes (unter Verwendung monatlicher Verteilungen)
- **Ausgabe**: `data/test_final.csv`


DIESER Flow ist suboptimal. Es wurde erkannt, dass es noch Probleme gab, nach dem Mergen, deshalb wurde die Wetter Imputation nochmals danach ausgeführt, bzw müsste man nochmals prüfen, ob man die erste Imputation nicht einfach weglassen kann, da die Final_wetter_imputation.py alle Werte behandelt. 

Auch nicht gut ist, dass dieses alles für das testdatenset ist, also wären allgemeine Funktionen besser, die man einfach importiert und dann auch für das Training verwenden kann. Hier wäre eine allgemeine Umstrukturierung wesentlich besser, denn das ist etwas chaotisch derzeit, zwar klar getrennt voneinander, aber nicht "Reusable Components" gerecht. 


ANALYSIS SCripts: 

In dem Ordner data/AnalysisScripts befinden sich verschiedene Skripte zur Analyse von Daten. Es wäre jetzt zu langwierig darauf einzugehen, aber unter anderem WetterCodeAnalyzer, MissingValuesAnalyzer und DataAnalyzer, von denen einige der Funktionen zur Analyse auch genutzt wurden, aber es ist auch viel Zeug dabei, dass nicht genutzt wurde. Die Bilder wurden dann teilweise in analysis_results (ordner) gespeichert und auch csv Dateien, aber das ist alles relativ messy, da wurde auch viel probiert etc. Aber mit den Sachen kann man was machen, wenn man möchte, ist aber eher für neue Features genutzt worden, wie zum beispiel zu erkennen, dass LastDayOfMonth ein wichtiges Feature ist. Das ist aber auf jeden Fall DAS Tool gewesen, um neue Features zu erkennen und auch zu begründen (siehe Bilder in analysis_results). 


HIER DIE ERKLÄRUNG DER FUNKTIONEN in Neural_simple.py


# Neuronales Netzwerk für Umsatzvorhersagen

Dieses Python-Skript implementiert ein Deep-Learning-Modell zur Vorhersage von Umsätzen basierend auf Wetterdaten und anderen zeitbezogenen Merkmalen.

## Hauptkomponenten

- **create_callbacks()**: Richtet Early Stopping und Lernratenreduzierung ein, um Überanpassung während des Trainings zu verhindern.

- **create_product_features(df)**: Generiert umfangreiche Merkmalssätze, darunter:
  - Polynomiale Temperaturmerkmale
  - Wettercodekategorisierung
  - Produktspezifische Interaktionen
  - Zeitbezogene Merkmale (Wochentag, Monat, Jahreszeit)
  - Lag-Merkmale für Zeitreihenmodellierung

- **plot_history(history, product_metrics)**: Erstellt Visualisierungen des Trainingsverlaufs und der Leistungsmetriken nach Produktkategorie.

- **prepare_and_predict_umsatz_nn(df)**: Hauptfunktion, die:
  - Den Merkmalssatz erstellt
  - Daten in Trainings- und Validierungssätze aufteilt
  - Merkmale skaliert
  - Das neuronale Netzwerk konfiguriert und trainiert
  - Die Modellleistung insgesamt und nach Produktkategorie auswertet

- **main()**: Orchestriert die Datenpipeline:
  - Lädt und fusioniert Datensätze
  - Behandelt fehlende Werte
  - Trainiert das Modell
  - Zeigt Leistungsmetriken an
  - Generiert Visualisierung der Ergebnisse

Das neuronale Netzwerk verwendet mehrere Dense-Layer mit Dropout und Batch-Normalisierung zur Umsatzvorhersage, mit spezifischen Optimierungen für jede Produktkategorie im Datensatz.


Das Skript "LearningRate" nutzt de fakto die gleichen Set Ups, wie neural_simple.py aber geht iterativ konfigurierbare Intervalle von Learning rates durch, um eine ideale zu finden. Das kann man noch für Dropout Layer basteln, oder ein Framework dazu nutzen. 

Final Network ist im Prinzip eine Kopie von neural_simple.py, aber es nutzt dann die finalen Test Files, die vorher aufbereitet worden sind und hat den Train_Split nicht drin. Das ist also für die Abgabe. 


Hier die Erklärung zur Multiple Regression: 


## Hauptkomponenten

- **create_interaction_features(df)**: Erstellt Interaktionsmerkmale zwischen Produktkategorien und Basisfaktoren wie:
  - Wettermerkmale (Temperatur, Bewölkung, Wettercodes)
  - Zeitbezogene Merkmale (Wochentag, Monat, Jahreszeit)
  - Ereignismerkmale (Feiertage, Kieler Woche, Silvester)

- **prepare_and_predict_umsatz(df)**: Hauptfunktion, die:
  - Die Daten skaliert
  - Interaktionsmerkmale generiert
  - Daten in Trainings- und Validierungssätze aufteilt
  - Das lineare Regressionsmodell trainiert
  - Die Modellleistung insgesamt und nach Produktkategorie auswertet
  - Produktspezifische Gleichungen aus den gelernten Koeffizienten erstellt

- **print_product_equations(product_equations)**: Gibt die linearen Gleichungen für jede Produktkategorie in einem lesbaren Format aus, was Einblicke in die wichtigsten Einflussfaktoren bietet.

- **main()**: Orchestriert die Datenpipeline:
  - Lädt und fusioniert Datensätze
  - Analysiert Wettercodes und Winddaten
  - Imputiert fehlende Werte
  - Bereitet Merkmale vor
  - Trainiert das Modell
  - Zeigt Ergebnisse und produktspezifische Vorhersagegleichungen an


Final_Regression hat also die gleiche Aufgabe wie Final_Network.py ist also ein Klon von Multiple Regression für die Abgabe. 


