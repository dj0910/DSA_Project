# DSA Projekt
Dies ist ein Repository, welches für ein Projekt zur Prävention von Herz-Kreislauf-Erkrankungen des Moduls "Data Science and Analytics", genutzt wird.
Das Ziel ist es hierbei eine Präventionsstrategie für Herzkrankheiten zu entwickeln, die mit hoher Genauigkeit eine mögliche Erkrankung hervorsagen und die genauen Risikofaktoren analysieren kann.

# Inhaltsverzeichnis
1. [Repo-Navigation](#Repo-Navigation)
    1. [Übersicht](#Uebersicht)
    2. [Installationsanleitung](#Installationsanleitung)
2. [Projektbeschreibung](#Projektbeschreibung)
3. [Parameterbewertung](#Parameterbewertung)
    1. [Parameterübersicht](#Parameterubersicht)
    2. [Manuelle Bewertung](#Manuelle-Bewertung)
    3. [PCA](#PCA)
    4. [Fazit](#Fazit)
4. [Genutztes statistisches Modell](#Genutztes-statistisches-Modell)

# Repo-Navigation

## Übersicht
In unserem Repository sind 3 Überordner zu finden:
-  docs
Der  "docs"-Ordner wird für jegliche Dokumentation, die im Laufe des Projekts anfällt, genutzt. Aktuell befinden sich dort zum Einen die Projektskizze, die das Projekt beschreibt und die Ziele definiert, diese soll jedoch im Verlauf des Projekts von dieser README-Datei ersetzt werden. Zum Anderen ist dort auch eine HTML-Datei zu finden, die sämtliche Graphen, Code und Erklärungen zu dem Hauptdatensatz enthalten, um die Informationen auch ohne clonen des Repos einsehen zu können. Dafür muss diese lediglich gedownloaded und dann geöffnet werden. Zuletzt befindet sich hier auch noch eine "requirements"-Datei, in der sich alle Libraries, die benötigt werden, befinden.
-  resources
Hier befinden sich momentan zwei Datensätze, jeweils als bereinigte und unbereinigte Version, die wir als Basis für das Projekt nutzen.
Einer dieser Datensätze wird aktiv genutzt, während der andere, nämlich "heart_predictions.csv" verworfen wurde, da zu wenige Observations vorliegen, um ein zuverlässiges Modell zu erstellen.
-  scripts
In diesem Ordner befinden sich sämtliche Skripte des Projekts.
Unter Anderem sind hier die Datenbereinigung und -Speicherung, namens "data_cleaning_and_save.py" zu finden.
Weiterhin liegt hier auch das Skript "Heart_2020_exploration_Visualization.ipynb, welches für die leichtere Lesbarkeit in Form eines Jupyter-Notebooks vorliegt. In diesem Skript sind sämtliche Graphen und Informationen zu finden, die dann daraus in die PDF in dem "docs"-Ordner exportiert wurden.
Zuletzt liegt hier das Skript "pca_heart2020.py", in der der Datensatz nach den Hauptkomponenten analysiert wird.

## Installationsanleitung
**Voraussetzungen:**
* Python Version 3.11.9
* Jupyter Version 6.5.4

**Installation:**
1. Klonen Sie das Repository:
```bash
git clone https://github.com/dj0910/DSA_Project.git
```
2. Navigieren Sie zum Order
```bash
cd DSA_Project
```
3. Installieren Sie alle benötigten Packages
```bash
pip install -r requirements.txt
```

# Projektbeschreibung 
Im Rahmen des Wahlpflichtmoduls Data Science and Analytics an der Hochschule Mannheim beschäfigen wir uns als Team Data Dazzlers, bestehend aus fünf Studierenden, intensiv mit dem Thema der Prävention von Herzkrankheiten. Diese Krankheiten zählen weltweit zu den führenden Todesursachen, wobei laut Angaben der Weltgesundheitsorganisation (WHO) bis zu 18 Millionen Menschen jährlich an ihnen sterben. In Deutschland allein sind jährlich etwa 350.000 Todesfälle aufgrund von Herz-Kreislauf-Erkrankungen zu verzeichnen, wie durch das statistische Bundesamt belegt wird. 
Die Prävention dieser Erkrankungen spielt eine entscheidende Rolle im Gesundheitsmanagement. Hierbei kommen moderne Technologien und insbesondere Wearables wie Smartwatches ins Spiel. Diese ermöglichen es einem Individuum, seine Gesundheitsdaten kontinuierlich zu messen und auf Grundlage dieser Informationen entsprechend zu handeln. 
Das Team Data Dazzlers konzentriert sich darauf, eine Präventionsstrategie für Herzkrankheiten zu entwickeln und zu optimieren. Ein zentraler Ansatzpunkt dabei ist die Untersuchung 
physikalischer Risikofaktoren, die durch Wearables messbar sind. Durch die Analyse und Auswertung dieser Daten können frühzeitig potenzielle Risiken erkannt und präventive Maßnahmen eingeleitet werden. Hierbei kommt modernste Data-Science-Technologie zum Einsatz, um die Zusammenhänge zwischen den gemessenen physikalischen Parametern und dem Risiko für Herzkrankheiten zu erforschen und zu verstehen. Das Ziel ist es, präzise und individualisierte Empfehlungen für die Gesundheitsvorsorge zu entwickeln, die auf den individuellen Risikoprofilen der Nutzer basieren.

**Konkrete Ziele des Projekts:**
- Prävention und Früherkennung von Herz-Kreislauf-Erkrankungen
- Vermeidung/Aufklärung von Volkskrankheiten
- Sensibilisierung des Bewusstseins für die eigene Gesundheit

# Parameterbewertung

## Parameterübersicht
Hier sind nochmal alle vorhandenen Parameter aufgelistet.
| Parametername  | Datenart      |
| :-------------:  |:-------------:|
| Herzerkrankung       | Kategoriell     |
| BMI       | Nummerisch     |
| Rauchen       | Kategoriell     |
|Alkoholkonsum|Kategoriell|
|Schlaganfall|Kategoriell|
|physische Gesundheit|Numerisch|
|mentale Gesundheit|Numerisch|
|Schwierigkeiten beim Laufen|Kategoriell|
|Geschlecht|Kategoriell|
|Alterskategorie|Kategoriell|
|Ethnie|Kategoriell|
|Diabetes|Kategoriell|
|physische Aktivität|Kategoriell|
|generelle Gesundheit|Kategoriell|
|Schlafdauer|Numerisch|
|Asthma|Kategoriell|
|Nierenerkrankung|Kategoriell|
|Hautkrebs|Kategoriell|
## Manuelle Bewertung
Diese Parameter wurden durch die Datenalayse und -visualisierung manuell als relevante Risikofaktoren eingestuft.
| Parametername  | Datenart      |
| :-------------:  |:-------------:|
| Alterkategorie       | Kategoriell     |
| Schlafdauer       | Nummerisch     |
| Nierenerkrankung       | Kategoriell     |
|generelle Gesundheit|Kategoriell|
|Schlaganfall|Kategoriell|

## PCA
Die PCA-Bewertung wurde vorbereitet und das Skript wurde geschrieben, allerdings steht hier die genaue Auswertung und das Fine-Tuning noch aus, weswegen hierzu noch keine abschließende Aussage getroffen werden kann. Dies wird jedoch als nächster Schritt im Projekt anvisiert. Hauptsächlich soll PCA genutzt werden, um die manuelle Bewertung der Attribute (s.o.) zu bestätigen oder zu widerlegen.

## Fazit
Im folgenden wird ein vorläufiges Fazit gezogen, da ein endgültiges Fazit erst mit dem Ende des Projekts gezogen werden kann.
Weiterhin fehlen zur Endbewertung auch noch andere Konkurrenzmodelle zur logistischen Regression die vielleicht noch erprobt werden und die Bewertung mithilfe der PCA.
Mit den aktuellen Parametern Alterskategorie, Schlafdauer, Nierenerkrankung, generelle Gesundheit und Schlaganfall, die momentan die wichtigsten Paramter zur Berechnung des Risikos darstellen kann man bereits jetzt einen Rückschluss zu den Projektzielen ziehen.
Der wichtigste Punkt die einem hierbei ins Auge springt, ist, dass es immer Parameter geben wird, die nicht von Wearables oder Smartwatches gemessen werden können. Dadurch ist bereits jetzt ersichtlich, dass zusätzlich zu den Smartwatch-Paramtern noch eine Art Fragebogen existieren muss, mit dem der Betroffenen in Regelmäßigen Abständen befragt werden muss, allerdings kann diese Befragung über eine Anwendung der Smartwatch ablaufen, damit nicht noch ein zusätzliches Gerät von Nöten ist.
Zu diesen Parameter zählen bislang die Alterskategorie, die Nierenerkrankung, der Schlaganfall und die genrelle Gesundheit.
Die Schlafdauer jedoch könnte über die Smartwatch gesteuert und dokumentiert werden.
Damit wäre man in der Lage, durch dieselbe Anwendung, die auch die regelmäßigen Fragebögen steuert das Risiko berechnen zu lassen und damit die Projektziele größtenteils zu erfüllen.
Zum einen wird dadurch die Prävention und Früherkennung von Herz-Kreislauf-Erkrankungen gefördert, denn Alarmfunktionen bei zu hohem Risiko würden den Nutzer warnen, dass sein Risiko erhöht ist und ihn Aufforderung bewusst darauf zu achten und möglicherweise ärztliche Beratung aufzusuchen.
Dadurch wird der Nutzer automatisch für die eigene Gesundheit sensibilisiert und es werden indirekt auch Volkskrankheiten vermieden.

# Genutztes statistisches Modell
Als statistisches Modell wird die logistische Regression genutzt. Es wurde sich für dieses Modell entschieden, da der Großteil der Daten kategoriell ist und somit viele der im Unterricht vorgestellten Modelle nicht mehr mit den vorliegenden Daten zu vereinen waren.
Allerdings wurde sich noch nicht fest auf die logistische Regression geeinigt und im Verlauf des Projekts kann es durchaus sein, dass auch alternative Modelle noch betrachtet werden, vor Allem wenn die Ergebnisse mit der logistischen Regression nicht die gewünschte Genauigkeit erbringen.