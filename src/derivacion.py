"""Regla de derivación de mensajes y sus indicadores de negocio.

Completá las dos funciones. Solo NumPy. Verificá tu trabajo con:  python -m pytest -q tests/test_derivacion.py
"""
import numpy as np

REVISION = "revision_humana"


def derivar(proba, clases, umbral_fraude, umbral_confianza):
    """Decide el destino de cada mensaje a partir de las probabilidades del modelo.

    proba : array (n_mensajes, n_clases) con P(clase | mensaje); columnas en el orden de `clases`.
    clases : lista o array con los nombres de las clases (incluye "fraude").

    Regla, en este orden de prioridad:
    1. Si P(fraude) >= umbral_fraude, el destino es "fraude" (aunque no sea la clase más probable).
    2. Si no, y la probabilidad máxima >= umbral_confianza, el destino es la clase más probable.
    3. En cualquier otro caso, el destino es REVISION ("revision_humana").

    Devuelve un array de n_mensajes strings.
    """
    clases = np.asarray(clases)
    n_mensajes = proba.shape[0]
    
    # Identificar el índice correspondiente a la clase "fraude"
    idx_fraude = np.where(clases == "fraude")[0][0]
    p_fraude = proba[:, idx_fraude]

    # Calcular la probabilidad máxima y el índice de la clase más probable para cada mensaje
    p_max_idx = np.argmax(proba, axis=1)
    p_max = np.max(proba, axis=1)

    # Inicializar el array de destinos con el valor por defecto (REVISION)
    destinos = np.full(n_mensajes, REVISION, dtype=object)

    # Condición 2: Probabilidad máxima >= umbral_confianza
    mask_confianza = p_max >= umbral_confianza
    destinos[mask_confianza] = clases[p_max_idx[mask_confianza]]

    # Condición 1: P(fraude) >= umbral_fraude (tiene prioridad sobre la confianza general)
    mask_fraude = p_fraude >= umbral_fraude
    destinos[mask_fraude] = "fraude"

    return destinos


def evaluar_derivacion(y_real, destino):
    """Calcula los indicadores del negocio. Devuelve un dict con tres claves:

    - "cobertura": proporción de mensajes derivados automáticamente (destino distinto de REVISION).
    - "exactitud_automatica": entre los derivados automáticamente, proporción con destino == y_real.
    - "fraude_perdido": entre los mensajes cuyo y_real es "fraude", proporción derivada automáticamente
      a un equipo que no es "fraude". Los enviados a revisión humana no cuentan como perdidos.

    Los tres valores son proporciones entre 0 y 1 (float).
    """
    y_real = np.asarray(y_real)
    destino = np.asarray(destino)
    n_total = len(y_real)

    if n_total == 0:
        return {
            "cobertura": 0.0,
            "exactitud_automatica": 0.0,
            "fraude_perdido": 0.0
        }

    # Máscara para mensajes derivados automáticamente (no van a revisión humana)
    auto_mask = destino != REVISION
    n_auto = np.sum(auto_mask)

    # 1. Cobertura: proporción de mensajes derivados automáticamente
    cobertura = float(n_auto / n_total)

    # 2. Exactitud automática: entre los automáticos, cuántos coinciden con y_real
    if n_auto > 0:
        exactitud_automatica = float(np.sum(destino[auto_mask] == y_real[auto_mask]) / n_auto)
    else:
        exactitud_automatica = 0.0

    # 3. Fraude perdido: entre los mensajes cuyo y_real es "fraude", cuántos fueron derivados automáticamente a otro equipo
    real_fraude_mask = y_real == "fraude"
    n_real_fraude = np.sum(real_fraude_mask)

    if n_real_fraude > 0:
        # Casos que son fraude real y fueron derivados de forma automática pero a un destino distinto de "fraude"
        fraude_perdido_mask = real_fraude_mask & auto_mask & (destino != "fraude")
        fraude_perdido = float(np.sum(fraude_perdido_mask) / n_real_fraude)
    else:
        fraude_perdido = 0.0

    return {
        "cobertura": cobertura,
        "exactitud_automatica": exactitud_automatica,
        "fraude_perdido": fraude_perdido
    }