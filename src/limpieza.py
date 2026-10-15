"""Limpieza de texto para los mensajes de clientes de Cauce.

Completá las dos funciones. Podés usar solo la biblioteca estándar de Python (re, unicodedata).
Verificá tu trabajo con:  python -m pytest -q tests/test_limpieza.py
"""
import re
import unicodedata


def quitar_tildes(texto: str) -> str:
    """Devuelve el texto sin tildes ni diéresis: á→a, é→e, ü→u, ñ→n (también en mayúsculas).

    Pista: unicodedata.normalize("NFKD", ...) separa cada letra de su tilde, y
    unicodedata.combining(c) indica si un carácter es una tilde "suelta".
    """
    if not isinstance(texto, str):
        return ""
    # Descompone el texto en caracteres base y marcas combinadas (diacríticos)
    nfkd_form = unicodedata.normalize("NFKD", texto)
    # Filtra descartando los caracteres de combinación (tildes, diéresis, etc.)
    # Nota: la eñe (ñ) no se separa en n + tilde en NFKD, por lo que se preserva intacta.
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


def limpiar_texto(texto) -> str:
    """Normaliza un mensaje para vectorizarlo. Aplicá estas reglas en este orden:

    1. Si `texto` no es un str (por ejemplo NaN o None), devolver "".
    2. Pasar a minúsculas y quitar tildes.
    3. Reemplazar las URL (empiezan con http://, https:// o www.) por el token _url_.
    4. Reemplazar las direcciones de correo electrónico por _email_.
    5. Reemplazar los montos por _monto_. Un monto es "$", "gs" o "usd" (como palabra
       independiente, con punto opcional: "gs.") seguido, con o sin un espacio, de un número.
    6. Reemplazar los demás números por _num_. Un número puede tener puntos o comas
       internos: 15.000, 8.350,50 y 30.123.456 son un solo número.
    7. Reemplazar por un espacio todo carácter que no sea una letra de la "a" a la "z",
       un guion bajo o un espacio en blanco (así desaparecen signos y emojis).
    8. Dejar un solo espacio entre palabras y quitar los espacios del principio y del final.

    Ejemplos:
        limpiar_texto("¡Me cobraron $ 15.000 y NO los reconozco!")  -> "me cobraron _monto_ y no los reconozco"
        limpiar_texto("mi DNI es 30.123.456 😡")                    -> "mi dni es _num_"
    """
    if not isinstance(texto, str):
        return ""

    # 1 y 2. Pasar a minúsculas y quitar tildes
    texto = texto.lower()
    texto = quitar_tildes(texto)

    # 3. Reemplazar URLs (http://, https:// o www.) por _url_
    texto = re.sub(r'https?://\S+|www\.\S+', '_url_', texto)

    # 4. Reemplazar direcciones de correo electrónico por _email_
    texto = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '_email_', texto)

    # 5. Reemplazar los montos por _monto_
    # Un monto es $, gs, gs. o usd seguido opcionalmente de espacios y un número con formato numérico
    texto = re.sub(r'(?i)(?:\b(?:gs\.?|usd)|\$)\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?\b|(?:\b(?:gs\.?|usd)|\$)\s*\d+\b', '_monto_', texto)

    # 6. Reemplazar los demás números por _num_ (números con puntos o comas internos o simples)
    texto = re.sub(r'\b\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?\b|\b\d+[.,]\d+\b|\b\d+\b', '_num_', texto)

    # 7. Reemplazar por un espacio todo carácter que no sea una letra de la "a" a la "z", un guion bajo o un espacio en blanco
    texto = re.sub(r'[^a-z_ ]', ' ', texto)

    # 8. Dejar un solo espacio entre palabras y quitar los espacios del principio y del final
    texto = re.sub(r'\s+', ' ', texto).strip()

    return texto