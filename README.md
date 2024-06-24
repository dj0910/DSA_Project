# DSA Projekt
Dies ist ein Repository, welches für ein Projekt zur Prävention von Herz-Kreislauf-Erkrankungen des Moduls "Data Science and Analytics", genutzt wird.
Das Ziel ist es hierbei eine Präventionsstrategie für Herzkrankheiten zu entwickeln, die mit hoher Genauigkeit eine mögliche Erkrankung hervorsagen und die genauen Risikofaktoren analysieren kann.

# Inhaltsverzeichnis
1. [Repo-Navigation](#Repo-Navigation)
    1. [Übersicht](#Uebersicht)
    2. [Installationsanleitung](#Installationsanleitung)
2. [Projektbeschreibung](#Projektbeschreibung)
3. [Mögliche Lösungen für das Problem](#Mögliche-Lösungen-für-das-Problem)
4. [Projektziele](#Projektziele)
5. [Auswahl der Datenquelle](#Auswahl-der-Datenquelle)
    1. [Beschreibung der Datenquelle](#beschreibung-der-datenquelle)
6. [Parameterbewertung](#Parameterbewertung)
    1. [Parameterübersicht](#Parameterubersicht)
    2. [Manuelle Bewertung](#Manuelle-Bewertung)
    3. [PCA](#PCA)
    4. [Parameter-Fazit](#Parameter-Fazit)
7. [Architektur-Diagramm](#Architektur-Diagramm)
8. [Genutztes statistisches Modell](#Genutztes-statistisches-Modell)
9. [Haftungsausschluss](#Haftungsausschluss)
10. [Grenzen des Modells](#grenzen-des-modells)
11. [Fazit](#Fazit)
12. [Ausblick](#Ausblick)

# Repo-Navigation

## Übersicht
In unserem Repository sind 4 Überordner zu finden:
-  **docs**  
Der  "docs"-Ordner wird für jegliche Dokumentation, die im Laufe des Projekts anfällt, genutzt. Aktuell befinden sich dort zum Einen die Projektskizze, die das Projekt beschreibt und die Ziele definiert, diese soll jedoch im Verlauf des Projekts von dieser README-Datei ersetzt werden. Zum Anderen ist dort auch eine HTML-Datei zu finden, die sämtliche Graphen, Code und Erklärungen zu dem Hauptdatensatz enthalten, um die Informationen auch ohne clonen des Repos einsehen zu können. Dafür muss diese lediglich gedownloaded und dann geöffnet werden. Zuletzt befindet sich hier auch noch eine "requirements"-Datei, in der sich alle Libraries, die benötigt werden, befinden.
-  **resources**  
Hier befinden sich momentan zwei Datensätze, jeweils als bereinigte und unbereinigte Version, die wir als Basis für das Projekt nutzen.
Einer dieser Datensätze wird aktiv genutzt, während der andere, nämlich "heart_predictions.csv" verworfen wurde, da zu wenige Observations vorliegen, um ein zuverlässiges Modell zu erstellen.
-  **scripts**  
In diesem Ordner befinden sich sämtliche Skripte des Projekts.
Unter Anderem sind hier die Datenbereinigung und -Speicherung, namens "data_cleaning_and_save.py" zu finden.
Weiterhin liegt hier auch das Skript "Heart_2020_exploration_Visualization.ipynb, welches für die leichtere Lesbarkeit in Form eines Jupyter-Notebooks vorliegt. In diesem Skript sind sämtliche Graphen und Informationen zu finden, die dann daraus in die PDF in dem "docs"-Ordner exportiert wurden.
Zuletzt liegt hier das Skript "pca_heart2020.py", in der der Datensatz nach den Hauptkomponenten analysiert wird.
- **deprecated**  
In diesem Ordner befinden sich sämtliche Skripte und Dateien, die im Verlauf des Projekts genutzt worden sind, jedoch inzwischen nicht mehr relevant sind.

## Installationsanleitung
**Voraussetzungen:**
* Python Version 3.11.9
* Jupyter Version 6.5.4

**Installation:**
1. Klonen Sie das Repository:
```bash
git clone https://github.com/dj0910/DSA_Project.git
```
2. Navigieren Sie zu folgendem Ordner
```bash
cd DSA_Project
```
3. Installieren Sie alle benötigten Packages
```bash
pip install -r docs/requirements.txt
```

# Projektbeschreibung 
Im Rahmen des Wahlpflichtmoduls Data Science and Analytics an der Hochschule Mannheim beschäfigen wir uns als Team Data Dazzlers, bestehend aus fünf Studierenden, intensiv mit dem Thema der Prävention von Herzkrankheiten. Diese Krankheiten zählen weltweit zu den führenden Todesursachen, wobei laut Angaben der Weltgesundheitsorganisation (WHO) bis zu 18 Millionen[<sup>1</sup>](https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)) Menschen jährlich an ihnen sterben. In Deutschland allein sind jährlich etwa 350.000 Todesfälle[<sup>2</sup>](https://www.destatis.de/DE/Themen/Gesellschaft-Umwelt/Gesundheit/Todesursachen/_inhalt.html) aufgrund von Herz-Kreislauf-Erkrankungen zu verzeichnen, wie durch das statistische Bundesamt belegt wird. 
Die Prävention dieser Erkrankungen spielt eine entscheidende Rolle im Gesundheitsmanagement. Hierbei kommen moderne Technologien und insbesondere Wearables wie Smartwatches ins Spiel. Diese ermöglichen es einem Individuum, seine Gesundheitsdaten kontinuierlich zu messen und auf Grundlage dieser Informationen entsprechend zu handeln. 
Das Team Data Dazzlers konzentriert sich darauf, eine Präventionsstrategie für Herzkrankheiten zu entwickeln und zu optimieren. Ein zentraler Ansatzpunkt dabei ist die Untersuchung 
physikalischer Risikofaktoren, die durch Wearables messbar sind. Durch die Analyse und Auswertung dieser Daten können frühzeitig potenzielle Risiken erkannt und präventive Maßnahmen eingeleitet werden. Hierbei kommt modernste Data-Science-Technologie zum Einsatz, um die Zusammenhänge zwischen den gemessenen physikalischen Parametern und dem Risiko für Herzkrankheiten zu erforschen und zu verstehen. Das Ziel ist es, präzise und individualisierte Empfehlungen für die Gesundheitsvorsorge zu entwickeln, die auf den individuellen Risikoprofilen der Nutzer basieren.

# Mögliche Lösungen für das Problem
**1. Prävention und Früherkennung von Herz-Kreislauf-Erkrankungen**  
Für die Prävention und Früherkennung von Herz-Kreislauf-Erkrankungen nutzen wir ein innovatives Machine-Learning-Konzept, das auf wissenschaftlichen Datensätzen basiert. Diese Datensätze umfassen Informationen über Risikofaktoren, Krankheitsverläufe und Präventionsstrategien. Mit Hilfe von Machine-Learning analysieren wir diese Daten, um Muster und Zusammenhänge zu identifizieren, die auf ein erhöhtes Risiko für Herz-Kreislauf-Erkrankungen hinweisen können.
Der Algorithmus funktioniert wie ein Punktesystem, welches das individuelle Risiko einer Person für Herz-Kreislauf-Erkrankungen bewertet. Dieses System vergibt auf Basis der Trainingsdaten sowie der persönlichen Daten des Nutzers Punkte. Durch die Bewertung verschiedener Risikofaktoren wird ein Gesamtrisikowert ermittelt, der anzeigt, wie hoch die Wahrscheinlichkeit ist, dass eine Person in der Gegenwart, oder womöglich in der Zukunft an einer Herz-Kreislauf-Erkrankung leiden könnte.
Gleichzeitig können mithilfe von Wearables kontinuierlich Vitalwerte wie Herzfrequenz, Blutdruck und Aktivitätsniveau von Personen erfasst werden. Diese Daten werden in Echtzeit erfasst und könnten mit unserem Machine-Learning-Modell analysiert werden. Auf Grundlage dieser Analyse werden personalisierte Empfehlungen zur Verbesserung des Lebensstils und zur Prävention von Herz-Kreislauf-Erkrankungen generiert.
Diese Empfehlungen können je nach Umsetzbarkeit von den Betroffenen selbst oder mithilfe des eigenen behandelnden Arztes realisiert werden. So können Betroffene aktiv ihren Lebensstil anpassen und gesundheitsschädigende Verhaltensmuster vermeiden. Diese proaktive Herangehensweise ermöglicht es, Herz-Kreislauf-Erkrankungen effektiv zu verhindern oder zumindest ihre Entwicklung zu verlangsamen.

**2. Vermeidung/Aufklärung von Volkskrankheiten**  
Das Gesundheitssystem konzentriert sich immer mehr auf die kurative, sprich die heilende, als auf die präventive, sprich vorbeugende Medizin[<sup>3</sup>](https://www.pedocs.de/frontdoor.php?source_opus=10361). Dadurch erkranken Menschen, die durch präventive Maßnahmen eigentlich nicht erkranken hätten müssen.
Durch Projekte, die sich auf die Prävention von Krankheiten konzentrieren, kann also nicht nur dem Einzelnen geholfen werden, ein gesünderes und krankheitsfreies Leben zu führen, sondern auch der Bevölkerung als Gemeinschaft. Denn vor allem Volkskrankheiten, wie Diabetes, Adipositas und Herz-Kreislauf-Erkrankungen belasten nicht nur die Betroffenen, sondern auch unser Gesundheitssystem massiv[<sup>4</sup>](https://www.uni-paderborn.de/fileadmin/psychisch-stark-am-arbeitsplatz/pdf/BDP-Bericht-2012.pdf), denn diese Erkrankungen sind meist nur der Anfang, auf den dann Folgeerkrankungen oder Sekundärerkrankungen folgen, durch die die Menschen noch kränker werden.

**3. Sensibilisierung des Bewusstseins für die eigene Gesundheit**  
Dadurch, dass man in Arztpraxen nur noch schwer Termine bekommt und die Ärzte, wenn man dann mal einen Termin hat, unter massivem Zeitdruck stehen, fühlen sich viele Menschen im Gesundheitssystem übersehen und auch allein gelassen mit ihren Fragen und Sorgen.
Durch Projekte wie diese, bei denen Betroffene auf einfache und bequeme Art und Weise einen Überblick über die eigene Gesundheit erhalten können, wird ihnen die Angst genommen und auch eine Richtung vorgegeben, an der sie sich orientieren können. Es kann sich sehr befreiend und stärkend anfühlen, die Kontrolle über die eigene Gesundheit zu haben und nicht auf fremde Hilfe angewiesen zu sein. Durch die Einfachheit solcher Anwendungen und die einfache Integration in den Alltag der Menschen beschäftigen sich diese wahrscheinlich mehr und auch lieber mit ihrer Gesundheit, als wenn solche Projekte nicht existieren würden und dadurch wird eine allgemein gesündere Bevölkerung angestrebt und gefördert.

# Projektziele
**a. Grundlegende Prävention von Herzkreislauferkrankungen**  
Das Ziel ist es, eine ganzheitliche und effektive Präventionsstrategie für Herz-Kreislauf-Erkrankungen zu entwickeln. Dabei sollen nicht nur Risikofaktoren identifiziert und adressiert werden, sondern auch Maßnahmen zur Förderung eines gesunden Lebensstils und zur Reduzierung von Risikoverhalten ergriffen werden. Die Präventionsstrategie soll sowohl auf individueller Ebene als auch auf der Ebene der Gemeinschaft und der öffentlichen Gesundheit wirksam sein. Es sollen evidenzbasierte Interventionen entwickelt werden, die auf den spezifischen Bedürfnissen und Risikoprofilen der Zielgruppen basieren.

**b. Bewertung der Aussagekraft von Wearables in Bezug auf Prävention von Herz-Kreislauf-Erkrankungen**  
Das Ziel ist es, die Wirksamkeit und Zuverlässigkeit von Wearables zur Prävention von Herz-Kreislauf-Erkrankungen zu untersuchen und zu bewerten. Dabei sollen verschiedene Arten von Wearables, wie Smartwatches und Fitness-Tracker, auf ihre Eignung hin analysiert werden, relevante Gesundheitsdaten zu erfassen und präventive Maßnahmen zu unterstützen.

**c. Top 2-3 Risikofaktoren ausgeben**  
Das Ziel ist es, die wichtigsten Risikofaktoren für den Betroffenen zu identifizieren. Durch umfassende Datenanalysen und Machine-Learning-Modelle sollen die Risikofaktoren ermittelt werden, die den größten Einfluss auf die mögliche Entstehung und Entwicklung einer Herz-Kreislauf-Erkrankung haben. Dadurch kann der Betroffene sich gezielt auf diese Faktoren konzentrieren und ihnen entgegenwirken.

**d. Verlaufskurve für das weitere Risiko**  
Das Ziel ist es, eine prognostische Risikokurve für die zukünftige Entwicklung des Risikos von Herz-Kreislauf-Erkrankungen zu erstellen. Durch die Integration von historischen Gesundheitsdaten, aktuellen Gesundheitswerten und prädiktiven Modellen sollen individuelle
Risikoprofile erstellt werden.
Diese Risikokurven sollen kontinuierlich aktualisiert und verfeinert werden, um zu zeigen, wie sich Verbesserungen oder Verschlechterungen der Risikofaktoren auf den Verlauf des Gesamtrisikos über die Zeit auswirken.

# Auswahl der Datenquelle
Die Entscheidung über die Datenquelle orientierte sich an der „Checkliste für Datenakquise“ und es war wichtig, eine Mischung aus primären und sekundären Datenquellen zu erhalten. 
Leider gestaltete sich der Zugriff auf medizinische Daten schwierig, wodurch die Auswahl der Datensätze generell sehr eingeschränkt war. 
Dabei wurde berücksichtigt, möglichst viele Attribute in den Datensätzen zu haben, um viele Risikofaktoren einzubeziehen und unterschiedlich gewichten zu können. Es war auch wichtig, keine leeren Beobachtungen in den Datensätzen zu haben, da sie den Datensatz verunreinigen können.  
Auch ein Bias kann einen Datensatz verunreinigen, jedoch war es schwierig, Bias vollständig zu vermeiden. Im ausgewählten Datensatz gibt es einen Response-Bias, da es sich um eine Befragung handelt und nicht um wissenschaftlich erhobene Daten. 
Dieser Bias tritt auf, wenn die Antworten in einer Umfrage oder Studie systematisch von bestimmten Faktoren beeinflusst werden, was zu einer Verzerrung der Ergebnisse führen kann. 
In diesem Fall könnten persönliche Fragen zur sportlichen Aktivität, zum Gewicht und anderen Gewohnheiten die Befragten dazu bringen, unehrlich zu antworten und somit zu einem Response-Bias führen.  
Obwohl der Datensatz einen gewissen Bias aufweist, wurde er aufgrund seiner Aktualität (2023), seines Umfangs (über 300.000 Einträge) und der Vielzahl interessanter und gut messbarer Attribute ausgewählt.

## Beschreibung der Datenquelle
|  | Beschreibung     |
| :-------------  |:-------------|
| **Link**     | https://www.kaggle.com/code/georgyzubkov/heart-disease-exploratory-data-analysis/notebook     |
| **Datum**       | 14.05.2024     |
| **Erstellungsjahr**       | 2023     |
|**Art der Datensammlung**|Jährliche Telefonumfrage|
|**Quelle**|Kaggle|
|**Mögliche Arten von Bias**|Response-Bias|
|**Vorteile**|+ Viele Datensätze <Br> + Attribute|
|**Nachteile**|- Stark subjektive Attribute, wie „Sportliche Aktivität“ zum Beispiel|

# Parameterbewertung


## Parameterübersicht
Hier sind alle vorhandenen Parameter aufgelistet.
| Parametername  |Beschreibung| Datenart      |Trifft Bias zu?|
| :-------------:  |:-------------:|:-------------:|:-------------:|
| Herzerkrankung       |  Liegt eine Herz-Erkrankung vor?    | Kategoriell| Nein |
| BMI       |   Body-Mass-Index   |Nummerisch | Ja|
| Rauchen       | Raucht der Befragte?     | Kategoriell| Ja|
|Alkoholkonsum|Trinkt der Befragte?| Kategoriell| Ja|
|Schlaganfall|Liegt ein Schlaganfall in der Vergangeheit?| Kategoriell|Nein |
|physische Gesundheit|Wie oft im Monat fühlt der Befragte sich „physisch ungesund“?| Numerisch| Ja|
|mentale Gesundheit|Wie oft im Monat fühlt der Befragte sich „mental ungesund“?| Numerisch| Ja|
|Schwierigkeiten beim Laufen|Schwierigkeit, Treppen zu steigen.|Kategoriell | Ja|
|Geschlecht|Geschlecht|Kategoriell |Nein |
|Alterskategorie|Alters-Kategorie|Kategoriell |Nein |
|Ethnie|Ethnie| Kategoriell|Nein |
|Diabetes|Liegt eine Diabetes-Erkrankung vor?|Kategoriell |Nein |
|physische Aktivität|Wurde in den letzten 30 Tagen Sport gemacht?|Kategoriell |Ja |
|generelle Gesundheit|Wie „gesund“ fühlt sich der Befragte?|Kategoriell |Ja |
|Schlafdauer|Wie viele Stunden Schlaf?|Numerisch | Ja|
|Asthma|Liegt diagnostiziertes Asthma vor?|Kategoriell |Nein |
|Nierenerkrankung|Liegt eine Nieren-Erkrankung vor?| Kategoriell|Nein |
|Hautkrebs|Liegt diagnostizierter Hautkrebs vor?|Kategoriell |Nein |


## Manuelle Bewertung
Bevor Algorithmen zur Bewertung der Parameter genutzt werden, ist es wichtig erstmal ein Gefühl für die Abhängigkeiten und die Signifikanz der einzelnen Parameter zu bekommen. Deswegen wurden in der "Heart_2020_Exploration_Visualization"-Datei erstmal sämtliche Parameter univariativ, bivariativ und multivariativ analysiert. Nachdem die Analyse soweit abgeschlossen war, wurden sich alle Graphen genau angeschaut und die relevantesten in Bezug zu unseren Projektziele ermittelt. 
Diese relevanten Parameter und ihre Analyse wurden hier nochmal zur Verdeutlichung eingefügt.  
Im weiteren Verlauf des Projekts werden, sofern bei der algorithmischen Bewertung der Parameter keinen Anpassungen vorzunehmen sind, diese Parameter genutzt, um das Machine-Learning-Modell zu trainieren.

Folgende Parameter wurden durch die Datenanalyse und -visualisierung manuell als relevante Risikofaktoren eingestuft.
| Parametername  | Datenart      |
| :-------------:  |:-------------:|
| Alterskategorie       | Kategoriell     |
| Schlafdauer       | Nummerisch     |
| Nierenerkrankung       | Kategoriell     |
|generelle Gesundheit|Kategoriell|
|Schlaganfall|Kategoriell|

<Br>

**Alterskategorie:**  
Der Graph weist darauf hin, dass mit steigendem Alter die Wahrscheinlichkeit an einer Herzkrankheit zu erkranken um ein Vielfaches erhöht ist.
Da der Prozentsatz in der Alterskategorie "80 oder älter" mehr als das 22-Fache der Kategorie "18-24" ist, ist die Alterskategorie als wichtiger Parameter für die Bewertung des Risikos zu sehen.

![Alterskategorie-Graph](docs/AgeCategory.png)

**Schlafdauer:**  
Es ist eine wellenartige Form des Graphen zu erkennen, bei dem das Minimum bei 7 Stunden durchschnittlicher Schlafdauer liegt. Sowohl bei mehr als auch bei weniger Stunden Schlaf erhöht sich das Risiko an einer Herzkrankheit zu leiden stetig.
Da der Unterschied bei bis zu 12% liegt, viele andere Schlafdauern ebenfalls erhöhte Wahrscheinlichkeiten zu erkranken aufweisen und die Schlafdauer eine einfach messbare Variable ist, wird sie mit hoher Signifikanz für die Bewertung des Risikos versehen.

![Alterskategorie-Graph](docs/SleepTime_plot.png)

**Nierenerkrankung:**  
Es existieren circa viermal so viele Herzerkrankte, die an einer Nierenerkrankung leiden, wie Nichtherzerkrankte, die an einer Nierenerkrankung leiden.
Das deutet auf eine starke Korrelation zwischen Herzkrankheit und Nierenerkrankung hin, wodurch der Parameter "Nierenerkrankung" in Betracht gezogen werden kann.

![Alterskategorie-Graph](docs/KidneyDisease_plot.png)

**Generelle Gesundheit:**  
Die empfundene Gesundheit ist ein stark subjektiver Parameter. Jedoch ist der Unterschied zwischen dem besten und dem schlechtesten Gesundheitszustand so hoch, dass man hier trotzdem von einem signifikanten Parameter für die Bewertung des Risikos reden kann.
Wichtig zu beachten ist jedoch auch, dass möglicherweise Befragte "In Ordnung" oder "Schlecht" angegeben haben, gerade weil sie eine Herzerkrankung haben. Dieser mögliche Bias muss bei der späteren Bewertung in Betracht gezogen werden.

![Alterskategorie-Graph](docs/GenHealth_plot.png)

**Schlaganfall:**  
Es existieren circa fünfmal so viele Herzerkrankte, die einen Schlaganfall hatten, wie Nichtherzerkrankte, die einen Schlaganfall hatten.
Das deutet auf eine starke Korrelation zwischen Herzkrankheit und Schlaganfall hin, wodurch der Parameter "Schlaganfall" in Betracht gezogen werden kann.

![Alterskategorie-Graph](docs/Stroke.png)


## PCA
Die PCA-Bewertung wurde vorbereitet und das Skript wurde geschrieben, allerdings steht hier die genaue Auswertung und das Fine-Tuning noch aus, weswegen hierzu noch keine abschließende Aussage getroffen werden kann. Dies wird jedoch als nächster Schritt im Projekt anvisiert. Hauptsächlich soll PCA genutzt werden, um die manuelle Bewertung der Attribute (s.o.) zu bestätigen oder zu widerlegen.

## Parameter-Fazit
Im folgenden wird ein vorläufiges Fazit gezogen, da ein endgültiges Fazit erst mit dem Ende des Projekts gezogen werden kann.
Weiterhin fehlen zur Endbewertung auch noch andere Konkurrenzmodelle zur logistischen Regression die vielleicht noch erprobt werden und die Bewertung mithilfe der PCA.
Mit den aktuellen Parametern Alterskategorie, Schlafdauer, Nierenerkrankung, generelle Gesundheit und Schlaganfall, die momentan die wichtigsten Paramter zur Berechnung des Risikos darstellen kann man bereits jetzt einen Rückschluss zu den Projektzielen ziehen.
Der wichtigste Punkt die einem hierbei ins Auge springt, ist, dass es immer Parameter geben wird, die nicht von Wearables oder Smartwatches gemessen werden können. Dadurch ist bereits jetzt ersichtlich, dass zusätzlich zu den Smartwatch-Paramtern noch eine Art Fragebogen existieren muss, mit dem der Betroffenen in Regelmäßigen Abständen befragt werden muss, allerdings kann diese Befragung über eine Anwendung der Smartwatch ablaufen, damit nicht noch ein zusätzliches Gerät von Nöten ist.
Zu diesen Parameter zählen bislang die Alterskategorie, die Nierenerkrankung, der Schlaganfall und die genrelle Gesundheit.
Die Schlafdauer jedoch könnte über die Smartwatch gesteuert und dokumentiert werden.
Damit wäre man in der Lage, durch dieselbe Anwendung, die auch die regelmäßigen Fragebögen steuert das Risiko berechnen zu lassen und damit die Projektziele größtenteils zu erfüllen.
Zum einen wird dadurch die Prävention und Früherkennung von Herz-Kreislauf-Erkrankungen gefördert, denn Alarmfunktionen bei zu hohem Risiko würden den Nutzer warnen, dass sein Risiko erhöht ist und ihn Aufforderung bewusst darauf zu achten und möglicherweise ärztliche Beratung aufzusuchen.
Dadurch wird der Nutzer automatisch für die eigene Gesundheit sensibilisiert und es werden indirekt auch Volkskrankheiten vermieden.

# Architektur-Diagramm
In diesem Data-Science-Projekt zur Prävention von Herz-Kreislauf-Erkrankungen durchlaufen wir folgende Schritte: 
1. Zunächst werden die Gesundheitsdaten aus der CSV-Datei gesammelt und in ein geeignetes Format überführt.  
2. Anschließend werden die Daten auf fehlende Werte und Ausreißer überprüft und nötige Korrekturen und Transformationen vorgenommen. Die gesäuberten Daten werden zusätzlich abgespeichert. Um die spätere Modellbewertung zu ermöglichen, werden die Daten in Trainings- und Testdatensätze aufgeteilt. 
3. Mit den vorbereiteten Trainingsdaten trainieren wir ein Machine-Learning-Modell. Verschiedene Modelle können verglichen und dasjenige mit der besten Leistung ausgewählt werden. 
4. Das ausgewählte Modell wird anhand des Testdatensatzes evaluiert, um seine allgemeine Vorhersagegenauigkeit zu bestimmen. Geeignete Metriken werden verwendet, um die Modellleistung zu quantifizieren. 
5. Basierend auf den Ergebnissen des Modells interpretieren wir die Zusammenhänge zwischen den Risikofaktoren und der Erkrankungswahrscheinlichkeit. Das Modell kann dann eingesetzt werden, um neue Daten vorherzusagen oder weiterführende Erkenntnisse zu gewinnen.  

![Architektur-Diagramm](docs/architecture_diagramm.png)

# Genutztes statistisches Modell
Als statistisches Modell wird die logistische Regression genutzt. Es wurde sich für dieses Modell entschieden, da der Großteil der Daten kategoriell ist und somit viele der im Unterricht vorgestellten Modelle nicht mehr mit den vorliegenden Daten zu vereinen waren.
Allerdings wurde sich noch nicht fest auf die logistische Regression geeinigt und im Verlauf des Projekts kann es durchaus sein, dass auch alternative Modelle noch betrachtet werden, vor Allem wenn die Ergebnisse mit der logistischen Regression nicht die gewünschte Genauigkeit erbringen.


# Haftungssauschluss

# Grenzen des Modells

# Fazit

# Ausblick