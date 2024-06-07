import cv2
import dlib
import numpy as np
from pathlib import Path
from datetime import datetime
import pandas as pd

excel_generated = False

print('[INFO] Démarrage du système...')
print('[INFO] Importation des modèles pré-entrainés...')

# Ces trois fichiers sont des modèles pré-entrainés
pose_predictor_68_point = dlib.shape_predictor("C:/Users/hp/OneDrive - ESMT/Documents/INGC2/base de l'IA/shape_predictor_68_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("C:/Users/hp/OneDrive - ESMT/Documents/INGC2/base de l'IA/dlib_face_recognition_resnet_model_v1.dat")
face_detector = dlib.get_frontal_face_detector()
print('[INFO] Importation des modèles pré-entrainés...')

# Permet de dessiner les figures sur la photo pour délimiter le visage
def transform(image, face_locations):
    coord_faces = []
    for face in face_locations:
        rect = face.top(), face.right(), face.bottom(), face.left()
        coord_face = max(rect[0], 0), min(rect[1], image.shape[1]), min(rect[2], image.shape[0]), max(rect[3], 0)
        coord_faces.append(coord_face)
    return coord_faces

# Liste globale pour stocker les noms des visages détectés
face_names = []

def encode_face(image):
    # On va d'abord détecter les visages sur l'image grâce à cette fonction disponible dans dlib.
    # Cette fonction retourne une liste de coordonnées des visages sur l'image.
    face_locations = face_detector(image, 1)

    face_encodings_list = []

    for face_location in face_locations:
        # DETECT FACES
        # Cette fonction, sur chaque localisation sur le visage, place un point jusqu'à 68
        # et permet ainsi d'avoir une meilleure qualité de détection.
        shape = pose_predictor_68_point(image, face_location)

        # Génère un vecteur de dimension 128
        face_encodings_list.append(np.array(face_encoder.compute_face_descriptor(image, shape, num_jitters=1)))

    # Transformation des coordonnées de l'image
    face_locations = transform(image, face_locations)
    return face_encodings_list, face_locations

def easy_face_reco(frame, known_face_encodings, known_face_names):
    rgb_small_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ENCODING FACE
    face_encodings_list, face_locations_list = encode_face(rgb_small_frame)
    detected_names = []

    for face_encoding in face_encodings_list:
        if len(face_encoding) == 0:
            return np.empty((0))
        
        # COMPARAISON AVEC LES VISAGES CONNUS
        vectors = np.linalg.norm(known_face_encodings - face_encoding, axis=1)

        tolerance = 0.6

        result = []
        for vector in vectors:
            if vector <= tolerance:
                result.append(True)
            else:
                result.append(False)

        if True in result:
            first_match_index = result.index(True)
            name = known_face_names[first_match_index]
            
            # Ajouter le nom uniquement si la personne est détectée
            detected_names.append(name)
            # On dessine les rectangle sur le visage avec le nom en bas etc
    for (top, right, bottom, left), name in zip(face_locations_list, detected_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 2, bottom - 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

    # Retourner la liste des noms détectés
    return detected_names

# Structure de données pour la gestion de la présence avec l'heure d'arrivée
def update_presence(name):
    if not presence_data[name]['present']:
        presence_data[name]['present'] = True
        presence_data[name]['arrival_time'] = datetime.now().strftime('%H:%M:%S')
        print(f"{name} est présent à {presence_data[name]['arrival_time']}.")

def calculate_lateness(name, scheduled_time):
    if presence_data[name]['present']:
        actual_time = datetime.now().strftime('%H:%M:%S')
        scheduled_time = datetime.strptime(scheduled_time, '%H:%M:%S')
        actual_time = datetime.strptime(actual_time, '%H:%M:%S')

        if actual_time > scheduled_time:
            lateness = actual_time - scheduled_time
            print(f"{name} est en retard de {lateness}.")
        else:
            print(f"{name} est à l'heure.")

def check_absence():
    end_time = datetime.strptime("09:14:00", '%H:%M:%S')

    if datetime.now() > end_time and not processed_names:
        for name in known_face_names:
            if name not in processed_names:
                if not presence_data[name]['present']:
                    print(f"{name} est absent.")
                else:
                    print(f"{name} est présent.")

def save_to_excel(data):
    # Convertir le dictionnaire en DataFrame
    df = pd.DataFrame(list(data.items()), columns=['prenom_nom', 'info'])

    # Extraire les informations de chaque personne dans des colonnes séparées
    df[['status', 'retard']] = pd.DataFrame(df['info'].tolist(), index=df.index)

    # Ajouter une colonne 'Presence' basée sur la colonne 'status'
    df['Presence'] = df['status'].apply(lambda x: 'Présent' if x else 'Absent')

    # Supprimer la colonne 'info' et 'status' qui ne sont plus nécessaires
    df = df.drop(['info', 'status'], axis=1)

    # Afficher le DataFrame
    print('df : ', df)

    # Définir le chemin complet du fichier Excel
    excel_file_path = "C:/Users/hp/OneDrive - ESMT/Documents/INGC2/base de l'IA/presence_status.xlsx"

    # Enregistrer le DataFrame dans un fichier Excel
    df.to_excel(excel_file_path, index=False)
    print(f"Données enregistrées dans {excel_file_path}.")

    # Arrêter le système après l'enregistrement des données
    print('[INFO] Arrêt du système...')
    video_capture.release()
    cv2.destroyAllWindows()
    exit()  # Terminer le programme

# Fonction principale
if __name__ == '__main__':
    input_directory = "C:/Users/hp/OneDrive - ESMT/Documents/INGC2/base de l'IA/visages_connus"
    # Définir le chemin complet du fichier Excel
    excel_file_path = "C:/Users/hp/OneDrive - ESMT/Documents/INGC2/base de l'IA/presence_status.xlsx"

    scheduled_time = "08:00:00"
    
    # On importe les visages qui sont dans notre base (dossier 'visage_connus')
    print('[INFO] Importation des visages...')
    face_to_encode_path = Path(input_directory)

    # On crée une variable tableau qui va stocker tous les visages connus
    files = [file_ for file_ in face_to_encode_path.rglob('*.jpg')]

    for file_ in face_to_encode_path.rglob('*.png'):
        files.append(file_)

    # On crée une variable de type tableau qui va stocker les noms des personnes dont le visage est dans la base
    known_face_names = ['sophie', 'leslie', 'rimbe', 'sandrine']

    # Structure de données pour la gestion de la présence avec l'heure d'arrivée
    presence_data = {name: {'present': False, 'arrival_time': None} for name in known_face_names}
    
    # Structure de données pour stocker les noms déjà traités
    processed_names = set()

    # Ce tableau va stocker le des encodages de chaque visage
    known_face_encodings = []

    # On parcourt la liste des fichiers des visages pour ouvrir chacun d'eux
    for file_ in files:
        image = cv2.imread(str(file_))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Encodage de chaque fichier
        face_encoded = encode_face(image)[0][0]
        known_face_encodings.append(face_encoded)

    print('[INFO] Visages importés')
    print('[INFO] Démarrage Webcam...')
    video_capture = cv2.VideoCapture(0)
    print('[INFO] Webcam démarrée')
    print('[INFO] Détection...')

    while True:
        ret, frame = video_capture.read()

        # Utiliser la valeur retournée de easy_face_reco
        detected_names = easy_face_reco(frame, known_face_encodings, known_face_names)
        
        # Mettre à jour la liste globale face_names
        face_names.extend(detected_names)

        for name in known_face_names:
            if name in detected_names and name not in processed_names:
                update_presence(name)
                processed_names.add(name)

                # Appel à la fonction calculate_lateness une seule fois après avoir mis à jour la présence
                calculate_lateness(name, scheduled_time)

        # Appel à la fonction check_absence à la fin de chaque itération
        check_absence()

        # Après 14:00, générer le fichier Excel
        current_time = datetime.now().strftime('%H:%M:%S')
        end_time = "09:14:00"
        if current_time >= end_time and not excel_generated:
            excel_generated = True  # Pour s'assurer que le fichier Excel n'est généré qu'une fois
            print('presence data : ', presence_data)
            save_to_excel(presence_data)

        cv2.imshow('Logiciel de reconnaissance faciale', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
