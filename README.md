Description du logiciel



Le logiciel est un système de reconnaissance faciale conçu pour détecter le visage d'un 
étudiant de l’ESMT dans une vidéo, utilisant la webcam comme source d'images. À partir 
d'une base de données de visages préenregistrés, le logiciel identifie le nom associé au 
visage détecté pour enregistrer la présence de l'étudiant lors d'une séance de cours. Si 
l'étudiant arrive à l'heure, c'est-à-dire avant 08H00, il sera considéré comme présent. En cas 
de retard, le système enregistre la durée de retard en minutes. À 11H30, tous les étudiants 
non détectés sont considérés comme absents d’après le règlement intérieur en cours du 
jour.


Notes pour l'environnement Windows 


Sur Windows, vous devrez peut-être installer les éléments suivants en plus :
opencv-python : Une bibliothèque Python pour la vision par ordinateur qui peut être 
installée en utilisant pip install opencv-python.
CMake : Un outil de gestion de projet multiplateforme. Vous pouvez télécharger CMake 
depuis le site officiel (https://cmake.org/download/) et l'installer.
Visual Studio et l'extension pour C++ : Dlib nécessite un compilateur C++ pour être installé. 
Vous pouvez télécharger Visual Studio depuis le site officiel 
(https://visualstudio.microsoft.com/fr/) et installer l'extension C++ lors de l'installation.
Ces étapes supplémentaires sont nécessaires pour s'assurer que l'installation de dlib se 
déroule correctement sur un environnement Windows.

Bibliothèques à installer


Ces bibliothèques ont été installés et embarqués avec le projet :
- opencv
- dlib
- numpy
- imutils
- pillow
- pathlib
-datetime
-pandas


Important
Compatibles qu'avec des images .jpg et .png


Fonctionnalités
- Détection de visage
- Reconnaissance faciale
-vérifier la présence
-calcul retard

Les grandes lignes du code 
 -Importation des bibliothèques : Le script commence par l'importation des 
bibliothèques nécessaires, telles que OpenCV, Dlib, Pillow, NumPy, imutils, pathlib et 
datetime.
 -Chargement des modèles pré-entraînés : Il charge trois fichiers de modèles préentraînés pour la détection de visages, la prédiction des points de repère faciaux et 
l'encodage des visages.


 -Définition de fonctions utiles :
o transform(image, face_locations) : Transforme les coordonnées des visages 
détectés.
o encode_face(image) : Détecte les visages sur une image, extrait les 
encodages des visages et les coordonnées des points de repère.
o easy_face_reco(frame, known_face_encodings, known_face_names) : 
Effectue la reconnaissance faciale sur une trame vidéo, identifie les visages 
connus, et dessine des rectangles et des noms sur les visages détectés.
 - Initialisation des variables :
o input_directory : Chemin vers le répertoire contenant les visages connus.
o known_face_names : Liste des noms des personnes dont les visages sont 
connus.
o presence_data : Structure de données pour la gestion de la présence avec 
l'heure d'arrivée.
o processed_names : Ensemble pour stocker les noms déjà traités.
o known_face_encodings : Liste pour stocker les encodages des visages 
connus.
- Traitement des visages connus : Charge les visages connus à partir du répertoire 
spécifié, les encode et les stocke dans known_face_encodings.
- Démarrage de la webcam et détection en temps réel :
o Démarre la webcam.
o Capture chaque trame vidéo.
o Utilise la fonction easy_face_reco pour reconnaître les visages et mettre à 
jour la présence.
o Affiche la trame vidéo avec des rectangles et des noms.
o Définie l’état de l’étudiant détecté 
o Calcul le retard
o Enregistre les résultats dans un fichier Excel
Lorsque la touche 'q' est enfoncée, le script se termine en libérant la webcam et détruisant 
les fenêtres OpenCV.
- Arrêt du système
