import gradio as gr
import pdfplumber
import joblib
import unicodedata
import re
from openai import OpenAI

modelo = joblib.load("modelo.pkl")
vectorizador = joblib.load("vectorizador.pkl")
mlb = joblib.load("mlb.pkl")

cliente = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

SECCIONES = {
    "introduccion": ["INTRODUCCION", "INTRODUCCIÓN"],
    "planteamiento": ["PLANTEAMIENTO DEL PROBLEMA", "PROBLEMA", "DESCRIPCION DEL PROBLEMA", "DESCRIPCIÓN DEL PROBLEMA"],
    "objetivos": ["OBJETIVOS", "OBJETIVO GENERAL", "OBJETIVOS ESPECIFICOS", "OBJETIVOS ESPECÍFICOS"],
    "marco_teorico": ["MARCO TEORICO", "MARCO TEÓRICO", "FUNDAMENTACION TEORICA", "FUNDAMENTACIÓN TEÓRICA", "BASES TEORICAS", "BASES TEÓRICAS", "REVISION DE LITERATURA", "REVISIÓN DE LITERATURA"],
    "desarrollo": ["DESARROLLO", "METODOLOGIA", "METODOLOGÍA", "IMPLEMENTACION", "IMPLEMENTACIÓN", "PROCEDIMIENTO"],
    "resultado": ["RESULTADO", "RESULTADOS", "ANALISIS DE RESULTADOS", "ANÁLISIS DE RESULTADOS"],
    "conclusiones": ["CONCLUSION", "CONCLUSIONES", "CONCLUSIÓN"],
    "referencias": ["REFERENCIAS", "BIBLIOGRAFIA", "BIBLIOGRAFÍA", "FUENTES CONSULTADAS"]
}

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

def normalizar_texto(texto):
    texto = texto.upper()
    texto = unicodedata.normalize('NFD', texto)
    texto = texto.encode('ascii', 'ignore').decode('utf-8')
    return texto

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

def detectar_secciones(texto):
    encontradas = []
    faltantes = []
    texto_normalizado = normalizar_texto(texto)
    for seccion, variantes in SECCIONES.items():
        encontrada = False
        for variante in variantes:
            variante_normalizada = normalizar_texto(variante)
            if variante_normalizada in texto_normalizado:
                encontrada = True
                break
        if encontrada:
            encontradas.append(seccion)
        else:
            faltantes.append(seccion)
    return encontradas, faltantes

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
                variante_normalizada = normalizar_texto(variante)
                if variante_normalizada in linea:
                    nueva_seccion = seccion
                    break
            if nueva_seccion:
                break
        if nueva_seccion:
            seccion_actual = nueva_seccion
            if seccion_actual not in contenido_secciones:
                contenido_secciones[seccion_actual] = ""
        else:
            if seccion_actual:
                contenido_secciones[seccion_actual] += " " + linea
    return contenido_secciones

def contar_palabras(texto):
    texto = re.sub(r'[^A-Za-zÁÉÍÓÚáéíóúÑñ0-9 ]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto)
    palabras = texto.strip().split(" ")
    palabras = [p for p in palabras if len(p.strip()) > 0]
    return len(palabras)

def calcular_porcentaje(encontradas):
    return round((len(encontradas) / len(SECCIONES)) * 100, 2)

def generar_estado(porcentaje):
    if porcentaje == 100: return "Correcto"
    elif porcentaje >= 60: return "Incompleto"
    elif porcentaje >= 30: return "Deficiente"
    else: return "Mal estructurado"

def analizar_con_qwen(texto):
    try:
        respuesta = cliente.chat.completions.create(
            model="qwen3-1.7b",
            messages=[
                {"role": "system", "content": "Eres un evaluador académico. Analiza reportes de estadía y proporciona observaciones breves sobre estructura, claridad y contenido."},
                {"role": "user", "content": f"Analiza el siguiente reporte:\n\n{texto[:4000]}"}
            ],
            temperature=0.3
        )
        return respuesta.choices[0].message.content
    except Exception as e:
        return f"Error IA Local: {str(e)}"

def validar_documento(pdf):
    if pdf is None:
        return "Error", "0%", "", "", [], "No se cargó ningún archivo.", ""
    texto = extraer_texto(pdf.name)
    if texto is None:
        return "Error", "0%", "", "", [], "No se pudo leer el PDF.", ""
    texto_limpio = texto.strip()
    if len(texto_limpio) == 0:
        return "Error", "0%", "", "", [], "El documento está vacío.", ""
    if len(texto_limpio) < 50:
        return "Error", "0%", "", "", [], "El documento contiene muy poco texto.", ""

    encontradas, faltantes = detectar_secciones(texto)
    porcentaje = calcular_porcentaje(encontradas)
    estado = generar_estado(porcentaje)
    contenido_secciones = extraer_contenido_secciones(texto)

    analisis_palabras = []
    for seccion in SECCIONES.keys():
        contenido = contenido_secciones.get(seccion, "")
        cantidad = contar_palabras(contenido)
        minimo = MINIMO_PALABRAS[seccion]
        estado_palabras = "✅ Correcto" if cantidad >= minimo else "⚠️ Muy corto"
        analisis_palabras.append([seccion.replace("_", " ").upper(), cantidad, minimo, estado_palabras])

    X = vectorizador.transform([texto])
    prediccion = modelo.predict(X)
    etiquetas_ia = mlb.inverse_transform(prediccion)
    diagnostico_ml = ", ".join(etiquetas_ia[0]) if len(etiquetas_ia[0]) > 0 else "El documento cumple con los criterios analíticos de la IA."

    observaciones_qwen = analizar_con_qwen(texto)

    enc_str = "\n".join([f"🔹 {s.replace('_', ' ').upper()}" for s in encontradas]) if encontradas else "Ninguna"
    fal_str = "\n".join([f"🔸 {s.replace('_', ' ').upper()}" for s in faltantes]) if faltantes else "Ninguna"

    return estado, f"{porcentaje}%", enc_str, fal_str, analisis_palabras, diagnostico_ml, observaciones_qwen

tema_custom = gr.themes.Cyberpunk(
    primary_hue="cyan",
    secondary_hue="pink",
    neutral_hue="slate",
)

css = """
.gradio-container { background: linear-gradient(135deg, #0b0c10 0%, #1f2833 100%) !important; }
.titulo-principal { text-align: center; color: #66fcf1; text-shadow: 0 0 12px #66fcf1, 0 0 25px #66fcf1; font-family: 'Courier New', monospace; font-weight: 900; letter-spacing: 2px; }
.sub-principal { text-align: center; color: #45a29e; text-shadow: 0 0 5px #45a29e; font-family: 'Courier New', monospace; margin-bottom: 25px; font-weight: bold; }
"""

with gr.Blocks(theme=tema_custom, css=css) as interfaz:
    gr.Markdown("<h1 class='titulo-principal'>⚡ CYBER-CORE REPORT VALIDATOR v3.0 ⚡</h1>")
    gr.Markdown("<p class='sub-principal'>SISTEMA DE AUDITORÍA ESTRUCTURAL IMPULSADO POR IA LOCAL Y MACHINE LEARNING</p>")
    
    with gr.Row():
        with gr.Column(scale=1):
            archivo_input = gr.File(label="📂 ARRASTRA TU REPORTE ACADÉMICO (PDF)", file_types=[".pdf"])
            btn_analizar = gr.Button("🔥 INICIAR ESCANEO CUÁNTICO", variant="primary")
        
        with gr.Column(scale=2):
            with gr.Row():
                out_estado = gr.Textbox(label="📊 DIAGNÓSTICO ESTRUCTURAL", interactive=False)
                out_porcentaje = gr.Textbox(label="📈 INTEGRIDAD DEL CONTENIDO", interactive=False)
    
    with gr.Row():
        with gr.Column():
            out_encontradas = gr.TextArea(label="🔹 SECCIONES DETECTADAS", interactive=False, max_lines=8)
        with gr.Column():
            out_faltantes = gr.TextArea(label="🔸 SECCIONES FALTANTES", interactive=False, max_lines=8)
    
    gr.Markdown("### 📊 DESGLOSE CRÍTICO DE VOLUMEN (MÉTRICA DE PALABRAS)")
    out_tabla = gr.Dataframe(headers=["SECCIÓN ANALIZADA", "CONTEO REAL", "MÍNIMO REQUERIDO", "ESTATUS"], interactive=False)
    
    with gr.Row():
        with gr.Column():
            out_ml = gr.TextArea(label="🤖 ALERTAS SISTÉMICAS (MACHINE LEARNING)", interactive=False, max_lines=4)
        with gr.Column():
            out_qwen = gr.TextArea(label="🧠 EVALUACIÓN HEURÍSTICA (LLM COGNITIVO)", interactive=False, max_lines=12)
    
    btn_analizar.click(
        fn=validar_documento,
        inputs=[archivo_input],
        outputs=[out_estado, out_porcentaje, out_encontradas, out_faltantes, out_tabla, out_ml, out_qwen]
    )

interfaz.launch()