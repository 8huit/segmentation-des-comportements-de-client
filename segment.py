def maskOutliers(s):
    """
    Fonction qui retourne le mask qui retourne les lignes d'une Serie, correspondant aux outliers

    Parameters
    ----------
    s : pandas Series
        une Serie de valeurs numériques

    Returns
    -------
    mask : Series
        une Serie de booleens correspondant au masque qui retourne les valeurs atypiques/Outliers
    """
    # premier Quantile
    Q1 = s.quantile(0.25)
    # 3e quantile
    Q3 = s.quantile(0.75)
    # Interquantile
    IQR = Q3 - Q1
    
    # les outliers sont toutes les valeurs qui sont inférieures à Q1-1.5*IQR et supérieure à  Q3+1.5*IQR
    mask=(s < (Q1 - 1.5 * IQR)) |(s > (Q3 + 1.5 * IQR))
    return mask

def clean(df):
    # copie par valeur
    dfCleaned=df.copy(deep=True)
    # passage en majuscule de la description, du stock code et invoice No
    dfCleaned['InvoiceNo']=dfCleaned['InvoiceNo'].str.upper()
    dfCleaned['StockCode']=dfCleaned['StockCode'].str.upper()
    dfCleaned['Description']=dfCleaned['Description'].str.upper()
    
    #suppression des doublons (en tenant compte de l'ensemble des variables)
    maskDoublons=dfCleaned.duplicated()
    dfCleaned=df[~maskDoublons]
    
    maskConflictingUnitPrice=dfCleaned.duplicated(['InvoiceNo','StockCode','InvoiceDate','CustomerID','Quantity'],keep=False)
    #suppression des prix unitaires contradictoires pour un meme produit/client/date d'achat
    dfCleaned=dfCleaned[~maskConflictingUnitPrice]
    
    #suppression des achats sans CustomerID
    dfCleaned=dfCleaned[dfCleaned['CustomerID'].notnull()]
    
    # creation de la variable Prix total
    dfCleaned['TotalPrice']=dfCleaned['Quantity']*dfCleaned['UnitPrice']
    # creation des variables Annees et mois
    dfCleaned["Year"]=dfCleaned["InvoiceDate"].apply(lambda x: x.strftime('%Y'))
    dfCleaned["Month"]=dfCleaned["InvoiceDate"].apply(lambda x: x.strftime('%m'))
    dfCleaned["InvoiceDateSimple"]=df['InvoiceDate'].dt.date # date sans de la partie hh:mm:ss
    dfCleaned["InvoiceDateYM"]=df['InvoiceDate'].dt.strftime('%Y-%m')
    # creation de groupe de pays
    european_countries = [
        'United Kingdom', 'France', 'Netherlands', 'Germany', 'Norway', 'EIRE', 
        'Switzerland', 'Spain', 'Poland', 'Portugal', 'Italy', 'Belgium', 'Malta',
        'Lithuania', 'Iceland', 'Channel Islands', 'Denmark', 'Cyprus', 'Sweden',
        'Finland', 'Austria', 'Greece', 'Czech Republic', 'European Community'
    ]
    
    dfCleaned['CountryGroup'] = 'NotEurope'
    dfCleaned.loc[(dfCleaned.Country.isin(european_countries)), 'CountryGroup']  = 'Europe'
    dfCleaned.loc[(dfCleaned.Country == "United Kingdom"), 'CountryGroup']  = 'UK'
    return dfCleaned

import numpy as np

def computeRFM(df):
    # Creer la date de reference
    date_reference = df['InvoiceDate'].max()# + timedelta(days=1)
    
    # RFM groupé par client
    dfRFMc = df.groupby(['CustomerID']).agg({
            'Country': lambda x: pd.Series.mode(x)[0],
            'CountryGroup': lambda x: pd.Series.mode(x)[0],
            'InvoiceDate': lambda x: (date_reference - x.max()).days, #temps ecoulé depuis le dernier achat
            'InvoiceNo': 'count',
            'TotalPrice': 'sum'})
    
    # frequence sur 7 jours
    dfRFMc['Frequence7'] = (df.groupby('CustomerID').InvoiceNo.nunique() / 
        (date_reference - df.groupby('CustomerID').InvoiceDate.min()).dt.days * 7)
    with pd.option_context('mode.use_inf_as_na', True):
            mean_freq = dfRFMc['Frequence7'].mean(skipna=True)
    dfRFMc['Frequence7']=dfRFMc['Frequence7'].replace(np.inf, mean_freq) #remplacement des valeurs infinies par la moyenne
    
    # Renomme les colonnes 
    dfRFMc.rename(columns={'InvoiceDate': 'Recence',
                             'InvoiceNo': 'Frequence',
                             'TotalPrice': 'Montant'}, inplace=True)
    #joblib.dump(dfRFMc,'drive/MyDrive/Colab Notebooks/Datasets/dfRFMc.jbl.bz2')
    dfRFMc=dfRFMc.sort_values(['Recence','Frequence','Montant'], ascending=[True,False,False])
    
    # pour la frequence et le montant, le quantile de 80% correspond au seuil en dessous duquel la fréquence et le montant ne sont pas significatifs (score = 1)
    quantilesPareto = dfRFMc.quantile(q=[0.8])
    quantilesPareto['Recence']=dfRFMc['Recence'].quantile(q=0.2) #la recence la plus faible correspond au score le plus elevé (score = 2)
    quantilesPareto
    
    # Matrice de valeur
    dfRFMc['RScore']=np.where(dfRFMc['Recence']<=quantilesPareto['Recence'].iloc[0],2,1)
    dfRFMc['FScore']=np.where(dfRFMc['Frequence']>=quantilesPareto['Frequence'].iloc[0],2,1)
    dfRFMc['MScore']=np.where(dfRFMc['Montant']>=quantilesPareto['Montant'].iloc[0],2,1)
    dfRFMc['RFMScore']=dfRFMc['RScore'].map(str)+dfRFMc['FScore'].map(str)+dfRFMc['MScore'].map(str)
    dfRFMc.reset_index().sort_values('RFMScore',ascending=False)
    return dfRFMc

from sklearn import cluster

def trainModel(X):
    model = cluster.KMeans(n_clusters=6, init='k-means++')# pour avoir une initialisation aleatoire et unique
    model.fit(X)
    return model

def segment(X,model):
    y_pred=model.predict(X)
    result=pd.DataFrame({'customerid':X.index,'class':y_pred})
    return result

import pandas as pd
from sklearn.model_selection import train_test_split

print("starting main")
print("reading file...")
df=pd.read_excel("../Dataset/OnlineRetail.xlsx")
print("file loaded")
df=clean(df)
dfRFMc=computeRFM(df)
X=dfRFMc[['Recence','Frequence','Montant']]

X_std=X.copy()
X_std['Frequence']=X_std[~maskOutliers(X['Frequence'])]['Frequence']
X_std['Recence']=X_std[~maskOutliers(X['Recence'])]['Recence']
X_std['Montant']=X_std[~maskOutliers(X['Montant'])]['Montant']

maskIsNA=(X_std['Recence'].isna() | X_std['Frequence'].isna() | X_std['Montant'].isna())
X_train, X_test=train_test_split(X_std[~maskIsNA].values,test_size=0.3,random_state = 49)

cl=trainModel(X_train)

print(segment(X_std[~maskIsNA],cl))
print("end")