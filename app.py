import gradio as gr
import pdfplumber
import joblib
import unicodedata
import re

from openai import OpenAI

# ==========================================
# CARGAR MODELOS ML
# ==========================================

modelo = joblib.load("modelo.pkl")
vectorizador = joblib.load("vectorizador.pkl")
mlb = joblib.load("mlb.pkl")

# ==========================================
# CONEXION LM STUDIO
# ==========================================

cliente = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

# ==========================================
# SECCIONES Y VARIANTES
# ==========================================

SECCIONES = {

    "introduccion": [
        "INTRODUCCION",
        "INTRODUCCIÓN"
    ],

    "planteamiento": [
        "PLANTEAMIENTO DEL PROBLEMA",
        "PROBLEMA",
        "DESCRIPCION DEL PROBLEMA",
        "DESCRIPCIÓN DEL PROBLEMA"
    ],

    "objetivos": [
        "OBJETIVOS",
        "OBJETIVO GENERAL",
        "OBJETIVOS ESPECIFICOS",
        "OBJETIVOS ESPECÍFICOS"
    ],

    "marco_teorico": [
        "MARCO TEORICO",
        "MARCO TEÓRICO",
        "FUNDAMENTACION TEORICA",
        "FUNDAMENTACIÓN TEÓRICA",
        "BASES TEORICAS",
        "BASES TEÓRICAS",
        "REVISION DE LITERATURA",
        "REVISIÓN DE LITERATURA"
    ],

    "desarrollo": [
        "DESARROLLO",
        "METODOLOGIA",
        "METODOLOGÍA",
        "IMPLEMENTACION",
        "IMPLEMENTACIÓN",
        "PROCEDIMIENTO"
    ],

    "resultado": [
        "RESULTADO",
        "RESULTADOS",
        "ANALISIS DE RESULTADOS",
        "ANÁLISIS DE RESULTADOS"
    ],

    "conclusiones": [
        "CONCLUSION",
        "CONCLUSIONES",
        "CONCLUSIÓN"
    ],

    "referencias": [
        "REFERENCIAS",
        "BIBLIOGRAFIA",
        "BIBLIOGRAFÍA",
        "FUENTES CONSULTADAS"
    ]
}

# ==========================================
# MINIMO DE PALABRAS
# ==========================================

MINIMO_PALABRAS = {

    "introduccion": 150,
    "planteamiento": 150,
    "objetivos": 80,
    "marco_teorico": 500,
    "desarrollo": 800,
    "resultado": 300,
    "conclusiones": 150,
    "referencias": 20
}

# ==========================================
# NORMALIZAR TEXTO
# ==========================================

def normalizar_texto(texto):

    texto = texto.upper()

    texto = unicodedata.normalize(
        'NFD',
        texto
    )

    texto = texto.encode(
        'ascii',
        'ignore'
    ).decode('utf-8')

    return texto

# ==========================================
# EXTRAER TEXTO PDF
# ==========================================

def extraer_texto(pdf_file):

    texto = ""

    try:

        with pdfplumber.open(pdf_file) as pdf:

            for pagina in pdf.pages:

                contenido = pagina.extract_text()

                if contenido:

                    texto += contenido + "\n"

    except Exception:

        return None

    return texto

# ==========================================
# DETECTAR SECCIONES
# ==========================================

def detectar_secciones(texto):

    encontradas = []
    faltantes = []

    texto_normalizado = normalizar_texto(texto)

    for seccion, variantes in SECCIONES.items():

        encontrada = False

        for variante in variantes:

            variante_normalizada = normalizar_texto(
                variante
            )

            if variante_normalizada in texto_normalizado:

                encontrada = True
                break

        if encontrada:

            encontradas.append(seccion)

        else:

            faltantes.append(seccion)

    return encontradas, faltantes

# ==========================================
# EXTRAER CONTENIDO
# ==========================================

def extraer_contenido_secciones(texto):

    contenido_secciones = {}

    texto_normalizado = normalizar_texto(texto)

    lineas = texto_normalizado.splitlines()

    seccion_actual = None

    for linea in lineas:

        linea = linea.strip()

        if len(linea) == 0:

            continue

        nueva_seccion = None

        for seccion, variantes in SECCIONES.items():

            for variante in variantes:

                variante_normalizada = normalizar_texto(
                    variante
                )

                if variante_normalizada in linea:

                    nueva_seccion = seccion
                    break

            if nueva_seccion:

                break

        if nueva_seccion:

            seccion_actual = nueva_seccion

            if seccion_actual not in contenido_secciones:

                contenido_secciones[
                    seccion_actual
                ] = ""

        else:

            if seccion_actual:

                contenido_secciones[
                    seccion_actual
                ] += " " + linea

    return contenido_secciones

# ==========================================
# CONTAR PALABRAS
# ==========================================

def contar_palabras(texto):

    texto = re.sub(
        r'[^A-Za-zÁÉÍÓÚáéíóúÑñ0-9 ]',
        ' ',
        texto
    )

    texto = re.sub(
        r'\s+',
        ' ',
        texto
    )

    palabras = texto.strip().split(" ")

    palabras = [
        p for p in palabras
        if len(p.strip()) > 0
    ]

    return len(palabras)

# ==========================================
# CALCULAR PORCENTAJE
# ==========================================

def calcular_porcentaje(encontradas):

    total = len(SECCIONES)

    presentes = len(encontradas)

    porcentaje = (
        presentes / total
    ) * 100

    return round(porcentaje, 2)

# ==========================================
# GENERAR ESTADO
# ==========================================

def generar_estado(porcentaje):

    if porcentaje == 100:

        return "Correcto"

    elif porcentaje >= 60:

        return "Incompleto"

    elif porcentaje >= 30:

        return "Deficiente"

    else:

        return "Mal estructurado"

# ==========================================
# IA LOCAL QWEN
# ==========================================

def analizar_con_qwen(texto):

    try:

        respuesta = cliente.chat.completions.create(

            model="qwen3-1.7b",

            messages=[

                {
                    "role": "system",

                    "content": (
                        "Eres un evaluador académico. "
                        "Analiza reportes de estadía "
                        "y proporciona observaciones "
                        "breves sobre estructura, "
                        "claridad y contenido."
                    )
                },

                {
                    "role": "user",

                    "content": (
                        f"Analiza el siguiente reporte:\n\n"
                        f"{texto[:4000]}"
                    )
                }
            ],

            temperature=0.3
        )

        return respuesta.choices[0].message.content

    except Exception as e:

        return f"Error IA Local: {str(e)}"

# ==========================================
# VALIDAR DOCUMENTO
# ==========================================

def validar_documento(pdf):

    if pdf is None:

        return "No se cargó ningún archivo."

    texto = extraer_texto(pdf.name)

    if texto is None:

        return "No se pudo leer el PDF."

    texto_limpio = texto.strip()

    if len(texto_limpio) == 0:

        return "El documento está vacío."

    if len(texto_limpio) < 50:

        return (
            "El documento contiene muy poco texto."
        )

    # ==========================================
    # DETECTAR SECCIONES
    # ==========================================

    encontradas, faltantes = detectar_secciones(
        texto
    )

    porcentaje = calcular_porcentaje(
        encontradas
    )

    estado = generar_estado(
        porcentaje
    )

    # ==========================================
    # EXTRAER CONTENIDO
    # ==========================================

    contenido_secciones = (
        extraer_contenido_secciones(texto)
    )

    # ==========================================
    # ANALISIS PALABRAS
    # ==========================================

    analisis_palabras = []

    for seccion in SECCIONES.keys():

        contenido = contenido_secciones.get(
            seccion,
            ""
        )

        cantidad = contar_palabras(
            contenido
        )

        minimo = MINIMO_PALABRAS[
            seccion
        ]

        estado_palabras = "Correcto"

        if cantidad < minimo:

            estado_palabras = "Muy corto"

        analisis_palabras.append(
            (
                seccion,
                cantidad,
                minimo,
                estado_palabras
            )
        )

    # ==========================================
    # IA MACHINE LEARNING
    # ==========================================

    X = vectorizador.transform(
        [texto]
    )

    prediccion = modelo.predict(X)

    etiquetas_ia = mlb.inverse_transform(
        prediccion
    )

    # ==========================================
    # IA LOCAL QWEN
    # ==========================================

    observaciones_qwen = analizar_con_qwen(
        texto
    )

    # ==========================================
    # RESULTADO
    # ==========================================

    resultado = ""

    resultado += (
        "====================================\n"
    )

    resultado += (
        "VALIDADOR ESTRUCTURAL DE REPORTES\n"
    )

    resultado += (
        "====================================\n\n"
    )

    resultado += (
        f"Estado general: {estado}\n"
    )

    resultado += (
        f"Cumplimiento estructural: "
        f"{porcentaje}%\n\n"
    )

    # ==========================================
    # SECCIONES DETECTADAS
    # ==========================================

    resultado += (
        "SECCIONES DETECTADAS:\n\n"
    )

    for seccion in encontradas:

        resultado += f"✔ {seccion}\n"

    resultado += "\n"

    # ==========================================
    # SECCIONES FALTANTES
    # ==========================================

    resultado += (
        "SECCIONES FALTANTES:\n\n"
    )

    if len(faltantes) == 0:

        resultado += "Ninguna\n"

    else:

        for seccion in faltantes:

            resultado += f"✘ {seccion}\n"

    resultado += "\n"

    # ==========================================
    # CONTEO PALABRAS
    # ==========================================

    resultado += (
        "CONTEO DE PALABRAS:\n\n"
    )

    for (
        seccion,
        cantidad,
        minimo,
        estado_palabras
    ) in analisis_palabras:

        resultado += (
            f"{seccion} → "
            f"{cantidad} palabras "
            f"(mínimo {minimo}) "
            f"[{estado_palabras}]\n"
        )

    resultado += "\n"

    # ==========================================
    # DIAGNOSTICO ML
    # ==========================================

    resultado += (
        "DIAGNÓSTICO IA (ML):\n\n"
    )

    if len(etiquetas_ia[0]) == 0:

        resultado += (
            "La IA considera que "
            "el documento está completo.\n"
        )

    else:

        for etiqueta in etiquetas_ia[0]:

            resultado += f"- {etiqueta}\n"

    resultado += "\n"

    # ==========================================
    # OBSERVACIONES QWEN
    # ==========================================

    resultado += (
        "====================================\n"
    )

    resultado += (
        "OBSERVACIONES IA LOCAL (QWEN3)\n"
    )

    resultado += (
        "====================================\n\n"
    )

    resultado += observaciones_qwen

    return resultado

# ==========================================
# INTERFAZ
# ==========================================

interfaz = gr.Interface(

    fn=validar_documento,

    inputs=gr.File(
        label="Subir PDF"
    ),

    outputs=gr.Textbox(
        label="Resultado",
        lines=40
    ),

    title=(
        "Validador Estructural "
        "de Reportes Offline"
    ),

    description=(
        "Sistema inteligente para "
        "validar reportes académicos "
        "mediante Machine Learning "
        "e IA Generativa Local."
    )
)

# ==========================================
# EJECUTAR
# ==========================================

interfaz.launch()