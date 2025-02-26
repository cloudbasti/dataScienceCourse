# dataScienceCourse

# [Introduction to Machine Learning and Data Science]

## Repository Link
https://github.com/cloudbasti/dataScienceCourse


## Beschreibung des Projektes

In dem Projekt sollen Umsatzdaten einer Bäckerei vorhergesagt werden, aufgrund von vergangenen Umsatzdaten verschiedener Produkte unter Hinzuziehung von Wetterdaten, Events wie Kieler Woche und weiterer Features. 


# Modellleistungsanalyse

## Einführung

Dieses Projekt demonstriert die erfolgreiche Implementierung fortschrittlicher Regressionsmodelle zur präzisen Umsatzprognose. Unser Ansatz verwendet sowohl traditionelle Maschinenlernverfahren als auch Deep-Learning-Techniken, um komplexe Datenmuster zu erfassen.

Das Baseline-Modell mit linearer Regression erzielte eine starke Leistung mit einem R-Quadrat-Wert von 0,860 und einem Root Mean Squared Error (RMSE) von 48,67, was eine solide Grundlage für unsere Prognosefähigkeiten darstellt.

Auf dieser Basis haben wir eine neuronale Netzwerkarchitektur implementiert, die die Vorhersagegenauigkeit weiter verbesserte und konsistent R-Quadrat-Werte zwischen 0,88 und 0,89 erreichte. Dies stellt eine signifikante Verbesserung gegenüber dem Baseline-Modell dar und unterstreicht die Fähigkeit des neuronalen Netzwerks, komplexere, nicht-lineare Beziehungen in den Daten zu erfassen.


### Bestes Modell: Neuronales Netzwerk
* **R-Quadrat:** 0.88-0.89
* **MAPE nach Produktkategorie:**
  * Produkt 1: 19.2%
  * Produkt 2: 13.9%
  * Produkt 3: 22.0%
  * Produkt 4: 23.2%
  * Produkt 5: 15.6%
  * Produkt 6: 59.4%

### Technische Details
* **Architektur:** 128 → 64 → 32 → 1 Neuronen
* **Hyperparameter:**
  * Batch-Größe: 32
  * Lernrate: 0.0008
  * Epochen: 50 mit Early Stopping
* **Regularisierung:** L2(0.01) mit Dropout-Schichten (30%, 30%, 12%)

Dieses Projekt demonstriert die erfolgreiche Implementierung fortschrittlicher Regressionsmodelle zur präzisen Umsatzprognose. Unser Ansatz verwendet sowohl traditionelle Maschinenlernverfahren als auch Deep-Learning-Techniken, um komplexe Datenmuster zu erfassen.

Das Baseline-Modell mit linearer Regression erzielte eine starke Leistung mit einem R-Quadrat-Wert von 0,860 und einem Root Mean Squared Error (RMSE) von 48,67, was eine solide Grundlage für unsere Prognosefähigkeiten darstellt.

Auf dieser Basis haben wir eine neuronale Netzwerkarchitektur implementiert, die die Vorhersagegenauigkeit weiter verbesserte und konsistent R-Quadrat-Werte zwischen 0,88 und 0,89 erreichte. Dies stellt eine signifikante Verbesserung gegenüber dem Baseline-Modell dar und unterstreicht die Fähigkeit des neuronalen Netzwerks, komplexere, nicht-lineare Beziehungen in den Daten zu erfassen.

## Modellarchitektur

Unsere Implementierung des neuronalen Netzwerks verfügt über eine sorgfältig abgestimmte Architektur:

- Eingabeschicht verbunden mit 128 Neuronen mit ReLU-Aktivierung und L2-Regularisierung
- Batch-Normalisierung gefolgt von 30% Dropout für verbesserte Generalisierung
- Versteckte Schichten mit 64 und 32 Neuronen, beide mit ReLU-Aktivierung und L2-Regularisierung
- Zusätzliche Dropout-Schichten zur Vermeidung von Overfitting
- Lineare Ausgabeschicht, optimiert für Regressionsaufgaben
- Adam-Optimizer mit einer Lernrate von 0,0008

Diese Konfiguration balanciert Modellkomplexität mit Generalisierungsfähigkeit und führt zu robusten Vorhersagen, die traditionelle Ansätze der linearen Regression übertreffen.

Die Leistungsverbesserungen, die durch unser neuronales Netzwerkmodell erzielt wurden, demonstrieren den Wert von Deep-Learning-Techniken für diese spezielle Vorhersageaufgabe, besonders bei der Verarbeitung komplexer, multidimensionaler Datenbeziehungen.
