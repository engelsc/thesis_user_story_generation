
# Automatische User Story Generierung mithilfe von LLMs

Dieses Repository enthält die Implementierung und die Analysepipeline meiner Bachelorarbeit zum Thema "Tauglichkeit von Open-Source- und kommerziellen LLMs im Requirements Engineering – Eine experimentelle Vergleichsstudie". Die Arbeit fokussiert sich auf die automatische Erstellung von User Stories aus App-Reviews mittels verschiedener Large Language Models (LLMs) und bewertet deren Qualität.
Das Ziel der Arbeit war die experimentelle Untersuchung der Eignung von kommerziellen und Open-Source LLMs zur automatischen Generierung von User Stories im Bereich des Requirements Engineering.

## Inhalt des Repositories

- Hauptpipeline (`main_program.py`): Automatische Generierung und Evaluierung von User Stories.

- Datenquellen:

	- Rohdaten (`data_sources/raw_data.csv`)

	- Datenbereinigungs-Skript (`data_sources/data_cleaning.py`)

- Zusatzexperiment (`experiment_multiple.py`): Zusätzlicher Versuch mit erweiterten Anforderungen.

- Analyse-Skript (`analysis_pipeline.py`): Detaillierte Auswertung der generierten User Stories.

- Ergebnisse: Alle generierten Stories inklusive AQUSA-Defekte befinden sich im Ordner `_tmp`. Die Auswertungen und Plots im Ordner `_analysis`.

- AQUSA-Core: Integrierte Qualitätsbewertung basierend auf dem AQUSA-Core-Framework (inklusive Python 3.8 venv mit allen notwendigen Abhängigkeiten).


## Voraussetzungen

- Python 3.12.8 für die Hauptpipeline
- Python 3.8 (virtuelle Umgebung .venv3.8) für AQUSA-Core
- Benötigte Bibliotheken sind in der Datei requirements.txt aufgelistet
- OpenRouter API Key (falls Stories generiert werden sollen)


## Installation
1. Repository klonen
2. Abhängigkeiten installieren

	```
	pip install -r requirements.txt
 	```

4. (Optional) OpenRouter API Key als Umgebungsvariable setzen (`OPEN_ROUTER_API_KEY`)
6. Python 3.8 in `.venv3.8` installieren
7. AQUSA requirements installieren
	Mit aktiver `.venv3.8`Umgebung:

	```
	pip install -r aqusa-core/requirements.txt
 	```

## Ausführen der Skripte
- Hauptpipeline zur User Story Generierung:

	```
	python main_program.py
 	```

	Um vorab generierte Ergebnisse in `_tmp` zu nutzen, Zeilen 121-122 auskommentieren.
	Zur Generierung mit LLMs wird ein OpenRouter API Key benötigt und die Datenbereinung muss vorher ausgeführt werden.
	
- Datenbereinigung
  
  ```
  python data_sources/data_cleaning.py
  ```
	
- Zusatzexperiment (mehrere Anforderungen)
  
	```
	python experiment_multiple.py
	```
	
- Analyse der mit AQUSA bewerteten Stories
  ```
  python analysis_pipeline.py
  ```
		
## AQUSA-core
Dieses Projekt verwendet AQUSA-Core zur Qualitätsbewertung generierter User Stories (MIT-License).
https://github.com/RELabUU/aqusa-core

## Datensatz
Dieses Projekt nutzt den folgenden Datensatz:
https://www.kaggle.com/datasets/saloni1712/threads-an-instagram-app-reviews

Die verwendeten Daten unterliegen CC BY-ND 4.0.
