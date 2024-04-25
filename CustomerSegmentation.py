import sklearn
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import plotly.graph_objs as go
from plotly.offline import iplot
from streamlit_card import card
import joblib
from sklearn.decomposition import PCA


st.set_page_config(layout="wide")

df = pd.read_csv("data/Customer_data_final.csv")

menu = ["Business Understanding", "Cluster Understanding","Product Recommendation"]
choice = st.sidebar.selectbox('Menu', menu )
if choice == 'Business Understanding':
  
    data = pd.read_csv("data/Online-retail.csv")

    numberClient =df["CustomerID"].value_counts().sum()

    nbr_product = data['InvoiceNo'].nunique()
    country = data['Country'].nunique()

    st.header("", divider='rainbow')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("### Customers")
        st.write(numberClient)

    with col2:
        st.write("### Products")
        st.write(nbr_product)

    with col3:
        st.write("#### Country")
        st.write(country)
   
    # Agréger les données par jour de la semaine
    daily_transactions = df.groupby('Day_Of_Week')['Total_Transactions'].sum().reset_index()

    # Traduire les numéros de jour de la semaine en noms de jour
    day_names = {1: 'Lundi', 2: 'Mardi', 3: 'Mercredi', 4: 'Jeudi', 5: 'Vendredi', 6: 'Samedi', 7: 'Dimanche'}
    daily_transactions['Day_Of_Week'] = daily_transactions['Day_Of_Week'].map(day_names)

    # Créer le graphique avec Plotly Express pour les transactions totales par jour de la semaine
    fig1 = px.bar(daily_transactions, x='Day_Of_Week', y='Total_Transactions', title='Total Transactions by Day of Week')


    # Traduire les numéros de jour de la semaine en noms de jour
    df['Day_Of_Week'] = df['Day_Of_Week'].map(day_names)

    # Créer le barplot avec Plotly Express
    fig2 = px.bar(df, x='Day_Of_Week', y='Total_Spend', title='Total Spend by Day of Week',
                labels={'Total_Spend': 'Total Spend ($)', 'Day_Of_Week': 'Day of Week'})

    customer_counts = data['Country'].value_counts()

    # Création d'une carte pour afficher le nombre de customerid par pays
    fig3 = px.choropleth(data_frame=data,
                        locations=customer_counts.index,
                        locationmode='country names',
                        color=customer_counts.values,
                        color_continuous_scale='YlOrRd',  # Utilisation de l'échelle de couleurs YlOrRd
                        title='Nombre of Customer  by country')
    fig3.update_layout(geo=dict(showcoastlines=True))
    
    # Calcul du nombre d'occurrences de chaque description unique et tri
    description_counts = data['Description'].value_counts()

    # Obtenir les 30 descriptions les plus fréquentes
    top_30_descriptions = description_counts.head(30)

    # Création d'un diagramme à barres horizontales avec Plotly
    trace = go.Bar(
        x=top_30_descriptions.values,
        y=top_30_descriptions.index,
        orientation='h'
    )

    # Mise en page
    layout = go.Layout(
        title='Top 30 Most Frequent Descriptions',
        xaxis=dict(title='Number of Occurrences'),
        yaxis=dict(title='Description')
    )

    # Créer la figure
    fig4 = go.Figure(data=[trace], layout=layout)
    col1, col2,col3 = st.columns(3)

    
    # Exemple de graphique interactif : Scatter plot avec Plotly
    fig5 = px.scatter(df, x="Total_Transactions", y="Total_Spend")

   # Calcul des proportions de la variable "Is_UK"
    # Calcul des proportions de la variable "Is_UK"
    uk_counts = df["Is_UK"].value_counts(normalize=True) * 100

    # Création du diagramme circulaire interactif avec Plotly Express
    fig6 = px.pie(uk_counts, values=uk_counts.values, names=uk_counts.index,width=300, title="Pie Chart of Is_UK")

        
    # Exemple de calcul des totaux des dépenses et des transactions pour chaque cluster
    totals_df = df.groupby('cluster').agg({
        'Total_Transactions': 'sum',
        'Total_Spend': 'sum'
    }).reset_index()

    # Créer un graphique à barres pour chaque cluster
    fig9 = go.Figure()

    # Ajouter les barres pour le total des dépenses avec une couleur différente
    fig9.add_trace(go.Bar(
        x=totals_df["cluster"],
        y=totals_df["Total_Spend"],
        name='Total Spend',
        marker_color='rgb(58, 73, 166)',
        width=0.5  # Couleur pour les dépenses
    ))

  
    # Mise en forme du graphique
    fig9.update_layout(
        title='Total Transactions by Cluster',
        xaxis=dict(
            title='Cluster'
        ),
        yaxis=dict(
            title='Amount'
        ),
        barmode='group'
    )


 
    # Afficher la carte
    col1, col2, col3 = st.columns([1, 1, 2])  # Utilisez une largeur de 1 pour les deux premières colonnes et 2 pour la troisième

    # Affichage des graphiques dans la première ligne
    with col1:
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.plotly_chart(fig2, use_container_width=True)

    # Affichage des graphiques dans la deuxième ligne
    col4, col5 = st.columns([1, 1])  # Utilisez une largeur de 1 pour les deux colonnes

    with col4:
        st.plotly_chart(fig4, use_container_width=True)

    with col5:
        st.plotly_chart(fig5, use_container_width=True)

    with col2:
        # Laissez cette colonne vide pour que fig3 prenne toute la ligne
        pass

    with col3:
        st.plotly_chart(fig6, use_container_width=True)

    col6, col7 = st.columns([1, 1])  # Utilisez une largeur de 1 pour les deux colonnes

    with col6:
        st.plotly_chart(fig3, use_container_width=True)

    with col7:
        st.plotly_chart(fig9, use_container_width=True)

    


elif choice == 'Cluster Understanding':
    st.markdown("<h2 style='margin-bottom:0'>Compréhension des clusters</h2>", unsafe_allow_html=True)
    st.header("", divider='rainbow')

    model = joblib.load("data/Kmeans.pkl")  # Assurez-vous de remplacer "chemin_vers_votre_modele.pkl" par le chemin de votre modèle .pkl

    class_colors = {
        "Monthly High-Spenders": "green",
        "Occasional High-Spenders": "blue",
        "High-Spending Churn Risk": "orange",
        "Low-Spending Weekend Shoppers": "purple",
    }

    # Définition des descriptions pour chaque cluster
    cluster_descriptions = {
        "Monthly High-Spenders": "Ce groupe se caractérise par des clients qui effectuent des achats fréquents avec des montants de dépenses mensuelles élevés. Ils sont susceptibles d'être des clients fidèles et réguliers, contribuant significativement aux revenus de l'entreprise. ",
        "Occasional High-Spenders": "Ces clients réalisent des achats moins fréquents par rapport aux Monthly High-Spenders, mais avec des montants de dépenses élevés lorsqu'ils achètent. Ils peuvent être des clients qui profitent des soldes, des promotions spéciales ou des événements saisonniers pour effectuer des achats importants. Cibler ce groupe avec des offres spéciales lors de périodes propices peut augmenter leur fréquence d'achat.",
        "High-Spending Churn Risk": "Ces clients ont des montants de dépenses élevés mais présentent un risque élevé de churn (résiliation) ou de perte en tant que clients. Il peut être nécessaire d'investir dans des stratégies de rétention et de service à la clientèle pour maintenir leur engagement et les fidéliser.",
        "Low-Spending Weekend Shoppers": "Ce groupe se compose de clients qui effectuent des achats de faible montant principalement pendant les week-ends. Bien qu'ils ne dépensent pas autant que les autres groupes, leur fréquence d'achat régulière peut en faire une cible intéressante pour les promotions et les offres spéciales du week-end. Ils sont susceptibles d'être réceptifs aux offres de week-end et peuvent être fidélisés en offrant des réductions  lors de ces périodes.",
    }

    # Création de deux colonnes pour positionner les cartes sur la même ligne
    col1, col2 = st.columns(2)

    # Affichage des cartes pour chaque cluster
    for cluster, color in class_colors.items():
        if cluster == "High-Spending Churn Risk" or cluster == "Low-Spending Weekend Shoppers":
            col2.write(f"### {cluster}")
            col2.markdown(f'<div style="background-color:{color};padding:10px;border-radius:5px;">{cluster_descriptions[cluster]}</div>', unsafe_allow_html=True)
        else:
            col1.write(f"### {cluster}")
            col1.markdown(f'<div style="background-color:{color};padding:10px;border-radius:5px;">{cluster_descriptions[cluster]}</div>', unsafe_allow_html=True)

    # Affichage des curseurs pour sélectionner les données du client
    st.header("", divider='rainbow')

    st.markdown("<h2 style='margin-bottom:0'>Entrer les donnees de client :</h2>", unsafe_allow_html=True)

    col4, _, col5 = st.columns([1, 0.1, 1])  # Utilisation d'une colonne vide avec un ratio de largeur de 0.1 pour créer l'espace

    # Ajouter une marge à droite pour créer un espace entre les colonnes
    col4.markdown('<style>div.row-widget.stHorizontal { flex-direction: row-reverse; }</style>', unsafe_allow_html=True)

    with col4:
        days_since_last_purchase = st.slider("Jours depuis le dernier achat", 0, 1000)
        total_transactions = st.slider("Nombre total de transactions", 0, 1000)
        total_products_purchased = st.slider("Nombre total de produits achetés", 0, 500)
        total_spend = st.slider("Total des dépenses", 0.0, 1000.0)
        average_transaction_value = st.slider("Valeur moyenne de transaction", 0.0, 100.0)
        monthly_spending_mean = st.slider("Moyenne des dépenses mensuelles", 0.0, 100.0)
        monthly_spending_std = st.slider("Écart-type des dépenses mensuelles", 0.0, 100.0)
    
    col5.markdown('<style>div.row-widget.stHorizontal { flex-direction: row-reverse; }</style>', unsafe_allow_html=True)

    with col5:
        unique_products_purchased = st.slider("Nombre de produits uniques achetés", 0, 1000)
        average_days_between_purchases = st.slider("Nombre moyen de jours entre les achats", 0.0, 10.0)
        day_of_week = st.slider("Jour de la semaine", 0, 6)
        hour = st.slider("Heure de la journée", 0, 23)
        is_uk = st.slider("Est au Royaume-Uni", 0, 1)
        cancellation_frequency = st.slider("Fréquence d'annulation", 0.0, 1.0)
        cancellation_rate = st.slider("Taux d'annulation", 0.0, 1.0)

    spending_trend = st.slider("Tendance des dépenses", 0.0, 10.0)


    # Création d'un DataFrame avec les données du client
    client_data = pd.DataFrame({
        'DaysSinceLastPurchase': [days_since_last_purchase],
        'TotalTransactions': [total_transactions],
        'TotalProductsPurchased': [total_products_purchased],
        'TotalSpend': [total_spend],
        'AverageTransactionValue': [average_transaction_value],
        'UniqueProductsPurchased': [unique_products_purchased],
        'AverageDaysBetweenPurchases': [average_days_between_purchases],
        'DayOfWeek': [day_of_week],
        'Hour': [hour],
        'IsUK': [is_uk],
        'CancellationFrequency': [cancellation_frequency],
        'CancellationRate': [cancellation_rate],
        'MonthlySpendingMean': [monthly_spending_mean],
        'MonthlySpendingStd': [monthly_spending_std],
        'SpendingTrend': [spending_trend]
    })

    # Chargement des données des clients prétraitées et transformation PCA
    data_scaled = pd.read_csv("data/data_scaled.csv")
    pca = PCA(n_components=4)
    customer_data_pca = pca.fit_transform(data_scaled)
    customer_data_pca = pd.DataFrame(customer_data_pca, columns=['PC'+str(i+1) for i in range(pca.n_components_)])
    customer_data_pca.index = data_scaled.index
  # Chargement des fonctions du notebook
    def apply_pca(data, pca_components):
        """
        Apply PCA transformation to the input data
        Args:
            data (np.ndarray): Input data
            pca_components (np.ndarray): PCA transformation matrix
        Returns:
            np.ndarray: Principal component scores
        """
        return data @ pca_components.T

    def get_prediction(model, pca_scores):
        """
        Use the trained model to make predictions on the PCA scores
        Args:
            model: Trained model
            pca_scores (np.ndarray): Principal component scores
        Returns:
            Prediction result
        """
        # Make predictions using the trained model
        predictions = model.predict(pca_scores)
        return predictions
    # Application de la transformation PCA aux données du client
    client_data_pca = apply_pca(client_data, pca.components_)

    # Définir le mapping des clusters aux classes
    class_mapping = {
        0: "Monthly High-Spenders",
        1: "Occasional High-Spenders",
        2: "High-Spending Churn Risk",
        3: "Low-Spending Weekend Shoppers",
        # Ajoutez d'autres mappings si nécessaire
    }

    # Définir les couleurs pour chaque classe
    class_colors = {
        "Monthly High-Spenders": "green",
        "Occasional High-Spenders": "blue",
        "High-Spending Churn Risk": "orange",
        "Low-Spending Weekend Shoppers": "purple",
        "Unknown": "gray"
    }

    # Fonction de prédiction du cluster
    def predict_cluster():
        # Prédiction du cluster pour le client
        cluster_label = get_prediction(model, client_data_pca)[0]
        
        # Mapping du cluster à la classe prédite
        predicted_class = class_mapping.get(cluster_label, "Unknown")
        
        return predicted_class

    # Bouton de prédiction
    if st.button("Prédire la classe"):
        predicted_class = predict_cluster()
        
        # Récupérer la couleur correspondante à la classe prédite
        class_color = class_colors.get(predicted_class, "gray")
        
        
        st.markdown(
    f'<div style="border: 5px solid {class_color}; padding: 10px; text-align: center;">'
    f'<span style="color: {class_color}; font-weight: bold; font-size: 24px;">Le client appartient à la classe : {predicted_class}</span>'
    '</div>',
    unsafe_allow_html=True
)
        
elif choice == 'Product Recommendation':
     

        # Charger les données
        def load_data():
            # Charger les données des recommandations (remplacez "customer_data_with_recommendations.csv" par le nom de votre fichier)
            recommendations_df = pd.read_csv("data/customer_data_with_recommendations.csv")
            return recommendations_df

        # Charger les données de recommandations
        recommendations_df = load_data()

        # Créer une application Streamlit
        st.text("Système de Recommandation de Produits")
        st.header("", divider='rainbow')


        # Sélectionner le nombre de clusters
        # Liste des noms de clusters possibles
        cluster_names = ['Monthly High-Spenders', 'Occasional High-Spenders', 'High-Spending Churn Risk', 'Low-Spending Weekend Shoppers']

        # Sélectionner le nom du cluster
        selected_cluster_name = st.selectbox("Sélectionner le nom du cluster :", cluster_names)

        # Filtrer les recommandations pour le nombre de clusters sélectionné
        filtered_recommendations = recommendations_df[recommendations_df['cluster'] == selected_cluster_name]

        # Afficher les recommandations pour le nombre de clusters sélectionné
        if not filtered_recommendations.empty:
            st.write(filtered_recommendations)
        else:
            st.write(f"Aucune recommandation disponible pour {selected_cluster_name} clusters.")

        # Assurez-vous que top_products_per_cluster est correctement défini
        # Calculer les produits les plus vendus pour chaque cluster
        top_products_per_cluster = recommendations_df.groupby('cluster').head(10)

        # Filtrer les données pour le cluster sélectionné
        top_products_cluster = top_products_per_cluster[top_products_per_cluster['cluster'] == selected_cluster_name]

        # Créer une liste pour stocker les données à afficher dans le tableau
        table_data = []

        # Utiliser un ensemble pour stocker les StockCodes déjà affichés
        displayed_stockcodes = set()
        st.text(f"Top produits pour le cluster {selected_cluster_name} :")
        st.header("", divider='rainbow')



        # Récupérer les 3 meilleurs produits avec leur StockCode pour le cluster sélectionné
        for index, row in top_products_cluster.iterrows():
            if row['Rec1_StockCode'] not in displayed_stockcodes:
                table_data.append([row['Rec1_StockCode'], row['Rec1_Description']])
                displayed_stockcodes.add(row['Rec1_StockCode'])
            if row['Rec2_StockCode'] not in displayed_stockcodes:
                table_data.append([row['Rec2_StockCode'], row['Rec2_Description']])
                displayed_stockcodes.add(row['Rec2_StockCode'])
            if row['Rec3_StockCode'] not in displayed_stockcodes:
                table_data.append([row['Rec3_StockCode'], row['Rec3_Description']])
                displayed_stockcodes.add(row['Rec3_StockCode'])


        st.table(pd.DataFrame(table_data, columns=["StockCode", "Description"]))
