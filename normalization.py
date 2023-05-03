from sklearn.preprocessing import normalize

df = pd.read_csv('my_data.csv')

X = df[['feature1', 'feature2', 'feature3']]

X_normalized = normalize(X, norm='l1', axis=0)

df[['feature1', 'feature2', 'feature3']] = X_normalized

df.to_csv('my_normalized_data.csv', index=False)
