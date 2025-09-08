
# Application Streamlit : Prédiction d'ouverture de compte bancaire (Inclusion financière)
# Tout le code est commenté en français, ligne par ligne, avec des variables nommées en français.
# Les “pages” sont organisées en onglets pour un déploiement simple sur Streamlit.

# -----------------------------
# 1) Importations des bibliothèques
# -----------------------------
import os  # Chargeons os pour manipuler les chemins de fichiers
import io  # Chargeons io pour gérer des flux mémoire (téléchargements)
import pickle  # Chargeons pickle pour sérialiser/désérialiser le modèle (utilisé pour export)
import numpy as np  # Chargeons numpy pour le calcul scientifique
import pandas as pd  # Chargeons pandas pour manipuler les tableaux de données
import plotly.express as px  # Chargeons plotly.express pour des graphiques interactifs
import plotly.graph_objects as go  # Chargeons plotly.graph_objects pour des graphiques personnalisés
import streamlit as st  # Chargeons Streamlit pour construire l'interface web
from imblearn.over_sampling import SMOTE  # Chargeons SMOTE pour équilibrer des classes minoritaires
from imblearn.pipeline import Pipeline as ImbPipeline  # Chargeons Pipeline (imblearn) pour chaîner étapes + SMOTE
from sklearn.compose import ColumnTransformer  # Chargeons ColumnTransformer pour appliquer des traitements par colonne
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # Chargeons OneHotEncoder/StandardScaler pour encoder/scaler
from sklearn.model_selection import train_test_split  # Chargeons train_test_split pour séparer train/test
from sklearn.ensemble import RandomForestClassifier  # Chargeons RandomForestClassifier pour modéliser la cible
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  # Chargeons les métriques utiles
from sklearn.utils import Bunch  # Chargeons Bunch pour structurer des retours lisibles
import streamlit.components.v1 as composants  # Chargeons components pour intégrer le rapport HTML de profiling

# >>> AJOUT : persistance compacte du modèle
import joblib  # Chargeons joblib pour (dé)serialiser en .pkl.gz

# -----------------------------
# 2) Configuration de la page Streamlit
# -----------------------------
st.set_page_config(  # Définissons la configuration générale de la page Streamlit
    page_title="Inclusion financière | Prédiction d'ouverture de compte",  # Titre de l’onglet navigateur
    page_icon="💳",  # Icône affichée dans l’onglet
    layout="wide"  # Largeur pleine pour profiter de l’espace
)

# -----------------------------
# 3) Chemins et constantes du projet
# -----------------------------
CHEMIN_FICHIER_DONNEES = "Financial_inclusion_dataset.csv"  # Définissons le chemin du fichier de données CSV
CHEMIN_PROFILING_HTML = "output.html"  # Définissons le chemin du rapport HTML de profiling (ydata-profiling)
# >>> AJOUT : chemin de l’artefact modèle compressé
CHEMIN_MODELE_PKL = "modele_inclusion.pkl.gz"  # Modèle compressé (gzip) persistant entre les runs

SEED_ALEATOIRE = 42  # Fixons une graine pour la reproductibilité
TAILLE_TEST = 0.2  # Définissons la taille du jeu de test (20%)
NB_ARBRES_RF = 300  # Fixons le nombre d’arbres du RandomForest pour un bon compromis vitesse/performance

# -----------------------------
# 4) Dictionnaires de renommage des colonnes (anglais -> français)
# -----------------------------
#    Chargeons un mapping clair pour renommer les colonnes en français (améliore la lisibilité côté EDA).
RENOMMAGE_COLONNES = {  # Définissons un dictionnaire pour renommer les colonnes en français
    "country": "pays",
    "year": "annee",
    "uniqueid": "identifiant_unique",
    "bank_account": "compte_bancaire",
    "location_type": "type_localisation",
    "cellphone_access": "acces_telephone",
    "household_size": "taille_menage",
    "age_of_respondent": "age_repondant",
    "gender_of_respondent": "genre_repondant",
    "relationship_with_head": "relation_chef_menage",
    "marital_status": "statut_matrimonial",
    "education_level": "niveau_education",
    "job_type": "type_emploi",
}

# -----------------------------
# 5) Fonctions utilitaires (chargement, renommage, préparation)
# -----------------------------

@st.cache_data(show_spinner=True)  # Mémorisons le résultat pour éviter de recharger à chaque interaction
def charger_donnees() -> pd.DataFrame:
    """Chargeons le fichier CSV officiel pour obtenir le jeu de données brut."""
    # Vérifions l'existence du fichier et signalons un message clair si absent
    if not os.path.exists(CHEMIN_FICHIER_DONNEES):  # Si le fichier n'existe pas
        st.error(f"Fichier introuvable : {CHEMIN_FICHIER_DONNEES}")  # Affichons une erreur utilisateur
        return pd.DataFrame()  # Retournons un DataFrame vide pour éviter une exception
    # Lisons le CSV dans un DataFrame pandas
    donnees = pd.read_csv(CHEMIN_FICHIER_DONNEES)  # Chargeons le jeu de données complet
    return donnees  # Retournons le DataFrame chargé

def renommer_colonnes_en_francais(donnees: pd.DataFrame) -> pd.DataFrame:
    """Renommons les colonnes techniques en français pour une meilleure lisibilité côté exploration."""
    # Copions les données pour éviter les effets de bord
    donnees_copie = donnees.copy()  # Créons une copie sûre du DataFrame d’entrée
    # Appliquons le dictionnaire de renommage
    donnees_copie = donnees_copie.rename(columns=RENOMMAGE_COLONNES)  # Renommons les colonnes
    return donnees_copie  # Retournons la version renommée

def colonnes_par_type(donnees_fr: pd.DataFrame) -> Bunch:
    """Identifions les colonnes numériques et catégorielles après renommage français."""
    # Sélectionnons les colonnes numériques par type
    colonnes_numeriques = donnees_fr.select_dtypes(include=[np.number]).columns.tolist()  # Récupérons les colonnes numériques
    # Sélectionnons les colonnes catégorielles par type
    colonnes_categorielles = donnees_fr.select_dtypes(exclude=[np.number]).columns.tolist()  # Récupérons les colonnes non numériques
    # Retournons un Bunch (objet pratique type dict) avec les deux listes
    return Bunch(numeriques=colonnes_numeriques, categorielles=colonnes_categorielles)

def preparer_cible(donnees_fr: pd.DataFrame) -> pd.Series:
    """Convertissons la cible 'compte_bancaire' (Yes/No) en binaire (1/0) en conservant le reste intact."""
    # Copions pour ne pas altérer l’original
    donnees_copie = donnees_fr.copy()  # Créons une copie pour travailler proprement
    # Convertissons la cible en 1/0 sans changer les autres colonnes
    cible_binaire = donnees_copie["compte_bancaire"].map({"Yes": 1, "No": 0})  # Mappage simple Yes->1, No->0
    return cible_binaire  # Retournons la série cible binaire

def separer_X_y(donnees_fr: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Séparons X (caractéristiques) et y (cible) à partir du DataFrame en français."""
    # Isolons y (cible)
    y_cible = preparer_cible(donnees_fr)  # Construisons la cible binaire 1/0
    # Retirons de X la cible et un identifiant qui n’apporte rien au modèle
    X_entree = donnees_fr.drop(columns=["compte_bancaire", "identifiant_unique"], errors="ignore")  # Supprimons la cible et l’identifiant
    return X_entree, y_cible  # Retournons X et y

def construire_transformeur(X: pd.DataFrame) -> ColumnTransformer:
    """Construisons un ColumnTransformer basé sur les colonnes présentes dans X (et pas le DF complet)."""
    types = colonnes_par_type(X)

    transformers = []
    if types.numeriques:
        transformers.append(("num", StandardScaler(), types.numeriques))
    if types.categorielles:
        # SMOTE exige du dense ; on garde OneHotEncoder en dense
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), types.categorielles))

    if not transformers:
        # Rien à transformer => levons une erreur explicite
        raise ValueError("Aucune colonne exploitable pour l'entraînement (ni numérique ni catégorielle).")

    return ColumnTransformer(transformers=transformers, remainder="drop")


@st.cache_resource(show_spinner=True)  # Mémorisons la ressource entraînée (modèle) entre interactions
def entrainer_modele(donnees_fr: pd.DataFrame) -> Bunch:
    """Entraînons un modèle RandomForest dans un pipeline : prétraitement -> SMOTE -> RandomForest."""
    # Séparons X et y
    X_entree, y_cible = separer_X_y(donnees_fr)  # Préparons les entrées et la cible
    # Construisons le transformeur de colonnes
    transformeur = construire_transformeur(X_entree)  # Créons le transformeur colonnes
    # Instancions le classifieur RandomForest
    classifieur = RandomForestClassifier(  # Définissons le modèle final
        n_estimators=NB_ARBRES_RF,  # Fixons le nombre d’arbres
        random_state=SEED_ALEATOIRE,  # Assurons la reproductibilité
        n_jobs=-1,  # Utilisons tous les cœurs disponibles
        class_weight=None  # La gestion du déséquilibre est confiée à SMOTE, pas aux poids
    )
    # Construisons un pipeline imbalanced-learn avec SMOTE entre le prétraitement et le classifieur
    pipeline = ImbPipeline(  # Chaînons les étapes dans l’ordre logique
        steps=[
            ("pretraitement", transformeur),  # Appliquons le transformeur (scaler + one-hot)
            ("smote", SMOTE(random_state=SEED_ALEATOIRE)),  # Équilibrons les classes dans l’espace transformé
            ("modele", classifieur),  # Appliquons le classifieur RF sur les données équilibrées
        ]
    )
    # Découpons en apprentissage/test
    X_app, X_test, y_app, y_test = train_test_split(  # Séparons apprentissage et test
        X_entree, y_cible, test_size=TAILLE_TEST, random_state=SEED_ALEATOIRE, stratify=y_cible  # Stratifier pour garder le ratio
    )
    # Entraînons le pipeline
    pipeline.fit(X_app, y_app)  # Ajustons l’ensemble des étapes sur les données d’apprentissage
    # Prédictions de contrôle sur le test
    y_pred = pipeline.predict(X_test)  # Prédictions classes
    y_proba = pipeline.predict_proba(X_test)[:, 1]  # Probabilités classe positive
    # Calculons des métriques clés
    exactitude = accuracy_score(y_test, y_pred)  # Calculons l’accuracy globale
    matrice = confusion_matrix(y_test, y_pred)  # Construisons la matrice de confusion
    rapport = classification_report(y_test, y_pred, output_dict=True)  # Générons un rapport détaillé (dict)
    # Retournons toutes les infos utiles dans un Bunch
    return Bunch(
        pipeline=pipeline,  # Renvoyons le pipeline complet prêt pour la prédiction
        X_test=X_test,  # Conserver X_test pour visualisations
        y_test=y_test,  # Conserver y_test pour métriques
        y_pred=y_pred,  # Conserver y_pred pour métriques
        y_proba=y_proba,  # Conserver y_proba pour courbes/threshold
        exactitude=exactitude,  # Score accuracy
        matrice=matrice,  # Matrice de confusion
        rapport=rapport  # Rapport détaillé (precision/recall/f1)
    )

# >>> AJOUT : helpers de persistance modèle
def sauver_modele_compresse(pipeline, chemin: str = CHEMIN_MODELE_PKL):
    """Sauvegardons le pipeline en .pkl.gz (joblib + gzip) pour persister entre les lancements."""
    joblib.dump(pipeline, chemin, compress=("gzip", 3))  # Compression niveau 3 = bon ratio/temps

def charger_modele_si_dispo(chemin: str = CHEMIN_MODELE_PKL):
    """Rechargeons le pipeline s’il existe déjà sur disque, sinon None."""
    if os.path.exists(chemin):
        try:
            return joblib.load(chemin)
        except Exception as e:
            st.warning(f"Impossible de charger le modèle existant ({e}). Il sera réentraîné.")
    return None

def dataframe_vers_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convertissons un DataFrame en CSV (bytes) pour permettre le téléchargement sans écrire sur disque."""
    # Créons un tampon mémoire texte
    tampon = io.StringIO()  # Ouvrons un buffer en mémoire
    # Écrivons le DataFrame en CSV dans le tampon
    df.to_csv(tampon, index=False)  # Exportons sans l’index
    # Convertissons en bytes
    return tampon.getvalue().encode("utf-8")  # Retournons les octets encodés en UTF-8

def dictionnaire_variables(donnees_fr: pd.DataFrame) -> pd.DataFrame:
    """Construisons un mini dictionnaire de données (nom, type, % manquants, exemples)."""
    # Préparons une liste pour collecter les méta-informations
    lignes = []  # Initialisons un conteneur vide
    # Parcourons chaque colonne pour en extraire les infos utiles
    for nom_colonne in donnees_fr.columns:  # Itérons sur les colonnes
        type_col = str(donnees_fr[nom_colonne].dtype)  # Récupérons le type pandas de la colonne
        manquants = int(donnees_fr[nom_colonne].isna().sum())  # Comptons le nombre de valeurs manquantes
        pct_manq = round(100 * manquants / len(donnees_fr), 2)  # Calculons le pourcentage de manquants
        exemple = donnees_fr[nom_colonne].dropna().unique()[:3]  # Prenons quelques exemples de valeurs
        lignes.append({  # Ajoutons une ligne d'information
            "variable": nom_colonne,
            "type": type_col,
            "valeurs_exemple": list(exemple),
            "manquants": manquants,
            "%_manquants": pct_manq
        })
    # Convertissons la liste en DataFrame structuré
    return pd.DataFrame(lignes)  # Retournons la table du dictionnaire

# -----------------------------
# 6) Chargement des données brutes, renommage et préparation
# -----------------------------
donnees_brutes = charger_donnees()  # Chargeons le fichier CSV pour obtenir le dataset d’origine
donnees_fr = renommer_colonnes_en_francais(donnees_brutes)  # Renommons les colonnes en français pour l’EDA

# Assurons-nous d'avoir une clé de session pour le modèle
if "modele_infos" not in st.session_state:
    st.session_state["modele_infos"] = None

# >>> AJOUT : Amorçage automatique du modèle (premier lancement)
if not donnees_fr.empty:
    with st.spinner("Initialisation du modèle…"):
        pipeline_disque = charger_modele_si_dispo()  # Tentons de charger un artefact existant
        if pipeline_disque is None:
            # Aucun artefact : entraînons maintenant et sauvegardons
            infos_init = entrainer_modele(donnees_fr)  # Entraînons le modèle initial
            sauver_modele_compresse(infos_init.pipeline)  # Persistance .pkl.gz pour les prochains runs
            st.session_state["modele_infos"] = infos_init  # Gardons aussi les métriques en session
            st.success(f"Modèle initial entraîné et sauvegardé (accuracy = {infos_init.exactitude:.3f}).")
        else:
            # Artefact déjà présent : utilisons-le immédiatement
            st.session_state["modele_infos"] = Bunch(pipeline=pipeline_disque)  # Pas de métriques tant qu’on ne réentraîne pas
            st.info("Modèle chargé depuis le disque (artefact compressé). "
                    "Cliquez sur **Réentraîner** pour générer de nouvelles métriques sur ce run.")

# -----------------------------
# 7) Barre d’entête (titre + rappel)
# -----------------------------
st.title("💳 Inclusion financière – Prédiction d'ouverture de compte bancaire")  # Affichons un titre clair
st.caption("Exploration, prétraitement, modélisation et prédiction – dataset d’inclusion financière.")  # Ajoutons un sous-titre

# -----------------------------
# 8) Organisation en onglets (pages logiques)
# -----------------------------
onglets = st.tabs([  # Créons des onglets pour structurer l’application
    "🏠 Accueil",            # Onglet 0 : Accueil
    "🔍 Exploration",        # Onglet 1 : Exploration des données
    "🧹 Prétraitement",      # Onglet 2 : Prétraitement (manquants, doublons, équilibrage)
    "🤖 Modélisation",       # Onglet 3 : Entraînement et évaluation du modèle
    "🎯 Prédiction",         # Onglet 4 : Interface de prédiction
    "📑 Profiling",          # Onglet 5 : Rapport ydata-profiling (HTML)
    "⬇️ Téléchargements",    # Onglet 6 : Export des artefacts
    "ℹ️ À propos"            # Onglet 7 : Contexte et objectifs
])

# -----------------------------
# 9) Onglet : Accueil
# -----------------------------
with onglets[0]:  # Entrons dans l’onglet Accueil
    st.subheader("Bienvenue 👋")  # Affichons un sous-titre convivial
    # Présentons le projet en quelques points clés
    st.markdown(  # Rédigeons une description du projet pour situer l’utilisateur
        """
        **Objet du projet :** analyser et prédire la probabilité qu’un individu possède un compte bancaire,
        afin d’éclairer des actions favorisant l’inclusion financière.

        **Ce que vous pouvez faire ici :**
        - Explorer le jeu de données (profils, distributions, corrélations)
        - Visualiser le prétraitement (manquants, doublons, valeurs atypiques)
        - Entraîner un modèle *Random Forest* (avec SMOTE pour traiter le déséquilibre)
        - Tester des prédictions sur des profils personnalisés
        - Consulter et télécharger le rapport de *profiling* et les artefacts (modèle, données préparées)
        """
    )
    # Affichons quelques KPI rapides
    col_a, col_b, col_c, col_d = st.columns(4)  # Créons une grille de 4 colonnes pour les indicateurs
    with col_a:  # Dans la 1ère colonne
        st.metric("Observations", value=f"{len(donnees_fr):,}".replace(",", " "))  # Affichons le nombre d’observations
    with col_b:  # Dans la 2ème colonne
        nb_vars = len(donnees_fr.columns)  # Comptons le nombre de variables
        st.metric("Variables", value=nb_vars)  # Affichons le nombre de colonnes
    with col_c:  # Dans la 3ème colonne
        part_yes = (donnees_fr["compte_bancaire"].eq("Yes").mean() * 100) if not donnees_fr.empty else 0  # Calculons le % Yes
        st.metric("% Comptes (Yes)", value=f"{part_yes:.1f}%")  # Affichons le pourcentage de comptes
    with col_d:  # Dans la 4ème colonne
        st.metric("Pays distincts", value=donnees_fr["pays"].nunique() if not donnees_fr.empty else 0)  # Affichons le nb de pays

# -----------------------------
# 10) Onglet : Exploration
# -----------------------------
with onglets[1]:  # Entrons dans l’onglet Exploration
    st.subheader("Aperçu des données")  # Affichons un sous-titre
    # Montrons les 10 premières lignes pour un aperçu rapide
    st.dataframe(donnees_fr.head(10), use_container_width=True)  # Présentons un échantillon des données
    # Ajoutons des stats descriptives numériques
    st.subheader("Statistiques descriptives (numériques)")  # Indiquons la section de stats
    st.dataframe(donnees_fr.describe(include=[np.number]), use_container_width=True)  # Résumons les colonnes numériques
    # Ajoutons une table sur les manquants
    st.subheader("Valeurs manquantes")  # Indiquons la section sur les valeurs manquantes
    tableau_manquants = pd.DataFrame({  # Construisons un tableau synthétique des manquants
        "variable": donnees_fr.columns,
        "nb_manquants": donnees_fr.isna().sum().values,
        "%_manquants": (100 * donnees_fr.isna().sum().values / len(donnees_fr)).round(2)
    })  # Fermons la construction du DataFrame
    st.dataframe(tableau_manquants, use_container_width=True)  # Affichons la table des manquants
    # Visualisation : distribution âge par pays (boxplot)
    st.subheader("Distribution de l’âge par pays")  # Annonçons la visualisation
    if not donnees_fr.empty:  # Vérifions que des données existent
        fig_age = px.box(donnees_fr, x="pays", y="age_repondant", points="outliers", title="Âge par pays")  # Créons un boxplot
        st.plotly_chart(fig_age, use_container_width=True)  # Affichons le graphique
    # Dictionnaire de données synthétique
    st.subheader("Dictionnaire de données (vue rapide)")  # Indiquons la section
    st.dataframe(dictionnaire_variables(donnees_fr), use_container_width=True)  # Affichons le dictionnaire rapide

# -----------------------------
# 11) Onglet : Prétraitement
# -----------------------------
with onglets[2]:  # Entrons dans l’onglet Prétraitement
    st.subheader("Contrôles de qualité")  # Affichons un sous-titre
    # Comptage de doublons
    nb_doublons = int(donnees_fr.duplicated().sum())  # Comptons les doublons ligne à ligne
    st.write(f"**Doublons détectés :** {nb_doublons}")  # Affichons le nombre de doublons
    # Affichons les colonnes numériques pour une détection visuelle d’outliers
    st.subheader("Valeurs atypiques – aperçu (boîtes à moustaches)")  # Indiquons la section outliers
    colonnes_num = donnees_fr.select_dtypes(include=[np.number]).columns.tolist()  # Récupérons les colonnes numériques
    for nom_col in colonnes_num:  # Parcourons chaque colonne numérique
        fig_box = go.Figure()  # Initialisons une figure Plotly
        fig_box.add_trace(go.Box(y=donnees_fr[nom_col], name=nom_col))  # Ajoutons une boîte à moustaches
        fig_box.update_layout(title=f"Boîte à moustaches – {nom_col}")  # Donnons un titre lisible
        st.plotly_chart(fig_box, use_container_width=True)  # Affichons le graphique

# -----------------------------
# 12) Onglet : Modélisation
# -----------------------------
with onglets[3]:  # Entrons dans l’onglet Modélisation
    st.subheader("Entraînement du modèle")  # Affichons un sous-titre

    # Bouton d’entraînement pour déclencher le calcul et (ré)écrire l’artefact
    if st.button("🔁 Entraîner / Réentraîner le modèle"):
        infos = entrainer_modele(donnees_fr)  # Entraînons et récupérons les objets utiles
        st.session_state["modele_infos"] = infos  # Conservons toutes les métriques en session
        sauver_modele_compresse(infos.pipeline)  # Sauvegardons l’artefact compressé
        st.success(f"Modèle (ré)entraîné et sauvegardé. Exactitude (accuracy) : {infos.exactitude:.3f}")

    # Affichage des métriques si disponibles
    infos = st.session_state.get("modele_infos", None)
    if infos is not None and hasattr(infos, "matrice"):
        # Affichons la matrice de confusion
        st.subheader("Matrice de confusion (jeu de test)")
        matrice = infos.matrice
        fig_cf = go.Figure(data=go.Heatmap(
            z=matrice,
            x=["Prédit: Non", "Prédit: Oui"],
            y=["Réel: Non", "Réel: Oui"],
            text=matrice,
            texttemplate="%{text}",
            colorscale="Blues"
        ))
        fig_cf.update_layout(title="Matrice de confusion")
        st.plotly_chart(fig_cf, use_container_width=True)

        # Affichons un résumé de classification (précision/rappel/F1)
        st.subheader("Rapport de classification")
        rapport_df = pd.DataFrame(infos.rapport).transpose()
        st.dataframe(rapport_df, use_container_width=True)
    elif infos is not None:
        st.info("Modèle chargé depuis le disque. Réentraînez pour afficher les métriques de ce run.")

# -----------------------------
# 13) Onglet : Prédiction
# -----------------------------
with onglets[4]:  # Entrons dans l’onglet Prédiction
    st.subheader("Prédire l’ouverture d’un compte bancaire")  # Affichons un sous-titre
    # Récupérons un modèle depuis la session si disponible
    infos = st.session_state.get("modele_infos", None)  # Lisons le modèle en session
    if infos is None:
        st.info("Le modèle n'est pas encore disponible. Vérifiez l'onglet **Modélisation**.")
    elif not hasattr(infos, "pipeline"):
        st.info("Le modèle chargé est incomplet. Réentraînez-le dans l’onglet **Modélisation**.")
    else:
        # Construisons un formulaire utilisateur
        with st.form("formulaire_prediction"):  # Créons un formulaire pour regrouper les champs
            col_g, col_d = st.columns(2)  # Créons deux colonnes pour une saisie lisible
            with col_g:  # Colonne gauche : attributs démographiques
                pays = st.selectbox("Pays", sorted(donnees_fr["pays"].dropna().unique()))
                age = st.slider("Âge du répondant", min_value=15, max_value=100, value=30, step=1)
                genre = st.selectbox("Genre", sorted(donnees_fr["genre_repondant"].dropna().unique()))
                statut = st.selectbox("Statut matrimonial", sorted(donnees_fr["statut_matrimonial"].dropna().unique()))
            with col_d:  # Colonne droite : autres attributs
                local = st.selectbox("Type de localisation", sorted(donnees_fr["type_localisation"].dropna().unique()))
                tel = st.selectbox("Accès au téléphone", sorted(donnees_fr["acces_telephone"].dropna().unique()))
                taille = st.slider("Taille du ménage", min_value=1, max_value=25, value=4, step=1)
                etude = st.selectbox("Niveau d’éducation", sorted(donnees_fr["niveau_education"].dropna().unique()))
                emploi = st.selectbox("Type d’emploi", sorted(donnees_fr["type_emploi"].dropna().unique()))
                relation = st.selectbox("Relation avec le chef de ménage", sorted(donnees_fr["relation_chef_menage"].dropna().unique()))
                annee = st.selectbox("Année d’enquête", sorted(donnees_fr["annee"].dropna().unique()))
            # Ajoutons un seuil de probabilité ajustable
            seuil = st.slider("Seuil de décision (probabilité pour classer 'Oui')", 0.05, 0.95, 0.50, 0.01)
            # Bouton de soumission du formulaire
            soumis = st.form_submit_button("Lancer la prédiction")
        # Si l’utilisateur a soumis le formulaire
        if soumis:
            # Construisons un DataFrame à une ligne avec les mêmes noms de colonnes (français) que X_entree
            entree_utilisateur = pd.DataFrame([{
                "pays": pays,
                "annee": annee,
                "type_localisation": local,
                "acces_telephone": tel,
                "taille_menage": taille,
                "age_repondant": age,
                "genre_repondant": genre,
                "relation_chef_menage": relation,
                "statut_matrimonial": statut,
                "niveau_education": etude,
                "type_emploi": emploi
            }])
            # Prédiction
            proba = infos.pipeline.predict_proba(entree_utilisateur)[:, 1][0]
            classe = int(proba >= seuil)
            # Rendu des résultats
            c1, c2 = st.columns(2)
            with c1:
                st.success(f"Probabilité d'avoir un compte : **{proba:.2%}**")
            with c2:
                libelle = "Compte probable (Oui)" if classe == 1 else "Compte peu probable (Non)"
                st.info(f"Décision (seuil {seuil:.0%}) : **{libelle}**")

# -----------------------------
# 14) Onglet : Profiling (rapport HTML)
# -----------------------------
with onglets[5]:  # Entrons dans l’onglet Profiling
    st.subheader("Rapport de profiling (ydata-profiling)")  # Affichons un sous-titre
    if not os.path.exists(CHEMIN_PROFILING_HTML):
        st.warning("Aucun rapport 'output.html' trouvé. Générez-le et placez-le à la racine.")
    else:
        with open(CHEMIN_PROFILING_HTML, "r", encoding="utf-8") as f:
            contenu_html = f.read()
        composants.html(contenu_html, height=800, scrolling=True)

# -----------------------------
# 15) Onglet : Téléchargements
# -----------------------------
with onglets[6]:  # Entrons dans l’onglet Téléchargements
    st.subheader("Export des artefacts")
    # Données renommées
    st.write("Téléchargez les **données renommées (français)** pour vos usages (EDA/archivage) :")
    st.download_button(
        label="⬇️ Télécharger le CSV (colonnes en français)",
        data=dataframe_vers_csv_bytes(donnees_fr),
        file_name="donnees_inclusion_fr.csv",
        mime="text/csv"
    )
    # Modèle compressé si disponible
    if os.path.exists(CHEMIN_MODELE_PKL):
        with open(CHEMIN_MODELE_PKL, "rb") as fbin:
            contenu_modele = fbin.read()
        st.download_button(
            label="⬇️ Télécharger le modèle (modele_inclusion.pkl.gz)",
            data=contenu_modele,
            file_name="modele_inclusion.pkl.gz",
            mime="application/octet-stream"
        )
    else:
        st.info("Aucun artefact modèle à télécharger pour l’instant (il sera généré au premier entraînement).")

# -----------------------------
# 16) Onglet : À propos
# -----------------------------
with onglets[7]:  # Entrons dans l’onglet À propos
    st.subheader("Contexte du projet")
    st.markdown(
        """
        **De quoi s’agit-il ?**  
        Projet d’inclusion financière visant à comprendre et **prédire** l’ouverture d’un compte bancaire.

        **Problème à résoudre :**  
        Faible taux de bancarisation et difficulté à **identifier les profils** susceptibles d’ouvrir un compte.

        **Objectifs :**  
        - Explorer les facteurs socio-démographiques impactant la possession d’un compte  
        - Construire un **modèle prédictif** fiable (RandomForest + SMOTE)  
        - Fournir une **interface** de prédiction et d’analyse décisionnelle
        """
    )
    st.caption("App Streamlit – variables Python en français, commentaires détaillés, profilage intégré.")
