import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def log_suma_exp(a, axis=1):
    """Calcula log(sum(exp(a))) a lo largo de `axis` sin overflow ni underflow.

    Truco: si m es el máximo de a, entonces log(sum(exp(a))) = m + log(sum(exp(a - m))).

    Ejemplo: log_suma_exp(np.array([[-1000.0, -1000.0]])) -> array([-999.3069...])
    """
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis=axis)


class NaiveBayesMultinomial(ClassifierMixin, BaseEstimator):
    """Naive Bayes multinomial con suavizado aditivo.

    Parámetros
    ----------
    alpha : float, por defecto 1.0
        Suavizado aditivo. Con alpha=1 es el suavizado de Laplace.

    Atributos que tiene que crear fit (mismos nombres que en scikit-learn)
    -------------------------------------------------------------------
    classes_ : array (n_clases,)
        Clases ordenadas, como devuelve np.unique(y).
    class_count_ : array (n_clases,)
        Cantidad de documentos de cada clase.
    feature_count_ : array (n_clases, n_terminos)
        Suma de los conteos de cada término en los documentos de cada clase.
    class_log_prior_ : array (n_clases,)
        log P(c) = log(documentos de la clase c / documentos totales).
    feature_log_prob_ : array (n_clases, n_terminos)
        log P(t | c) = log((conteo de t en c + alpha) / (total de términos en c + alpha * n_terminos)).
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        """Estima las probabilidades a partir de la matriz de conteos X y las etiquetas y. Devuelve self."""
        # Convertir a array de numpy o mantener compatibilidad con matrices dispersas
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Cantidad de documentos por clase
        self.class_count_ = np.array([np.sum(y == c) for c in self.classes_])
        
        # Probabilidad a priori en escala logarítmica: log P(c)
        n_docs = len(y)
        self.class_log_prior_ = np.log(self.class_count_ / n_docs)
        
        # Suma de conteos de términos por clase (feature_count_)
        # X soporta X[filas] y X.sum(axis=0) tanto para arrays como matrices dispersas
        self.feature_count_ = np.array([
            np.asarray(X[y == c].sum(axis=0)).flatten() 
            for c in self.classes_
        ])
        
        # Suavizado aditivo (Laplace si alpha=1) y cálculo de log P(t | c)
        smoothed_counts = self.feature_count_ + self.alpha
        smoothed_totals = smoothed_counts.sum(axis=1, keepdims=True)
        self.feature_log_prob_ = np.log(smoothed_counts) - np.log(smoothed_totals)
        
        return self

    def _log_verosimilitud_conjunta(self, X):
        """Devuelve log P(c) + sum_t x_t · log P(t | c) para cada documento y clase."""
        # Convertir a matriz densa si es dispersa de SciPy
        if hasattr(X, "toarray"):
            X_arr = X.toarray()
        else:
            X_arr = np.asarray(X)
            
        # Si la entrada es un vector unidimensional (1 sola muestra), convertirlo a 2D (1, n_features)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
            
        return X_arr @ self.feature_log_prob_.T + self.class_log_prior_

    def predict_log_proba(self, X):
        """log P(c | documento): la log-verosimilitud conjunta normalizada con log_suma_exp."""
        jll = self._log_verosimilitud_conjunta(X)
        log_prob_x = log_suma_exp(jll, axis=1)
        return jll - log_prob_x[:, np.newaxis]

    def predict_proba(self, X):
        """P(c | documento). Cada fila suma 1. Las columnas siguen el orden de classes_."""
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        """La clase con mayor probabilidad a posteriori para cada documento."""
        jll = self._log_verosimilitud_conjunta(X)
        return self.classes_[np.argmax(jll, axis=1)]


def terminos_mas_indicativos(modelo, vocabulario, n=10):
    """Devuelve {clase: [n términos]} con los términos que más empujan hacia cada clase.

    Para la clase c, el puntaje de un término t es
        log P(t | c) - promedio de log P(t | c') en las demás clases c'.
    Se devuelven los n términos de mayor puntaje, de mayor a menor.

    `vocabulario` es la lista de términos en el orden de las columnas de X
    (por ejemplo, CountVectorizer.get_feature_names_out()).
    """
    resultado = {}
    vocabulario = np.asarray(vocabulario)
    n_clases = len(modelo.classes_)
    
    for i, clase in enumerate(modelo.classes_):
        # Promedio de log P(t | c') en las demás clases c'
        out_classes = [j for j in range(n_clases) if j != i]
        promedio_otras = modelo.feature_log_prob_[out_classes].mean(axis=0)
        
        # Puntaje del término para la clase c
        puntajes = modelo.feature_log_prob_[i] - promedio_otras
        
        # Obtener los n índices de mayor puntaje
        top_indices = np.argsort(puntajes)[::-1][:n]
        resultado[clase] = vocabulario[top_indices].tolist()
        
    return resultado