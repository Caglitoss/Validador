# 📋 Validador Estructural de Reportes Offline con IA Local

Sistema inteligente para validar reportes académicos mediante Machine Learning e IA Generativa Local. Detecta secciones, valida estructura, cuenta palabras y proporciona retroalimentación detallada usando modelos ejecutados completamente en el equipo local.

Su objetivo es asegurar que los reportes académicos cumplan con estándares de estructura y contenido, manteniendo la privacidad de los datos al operar de forma completamente local.

---

# 🛠️ Requisitos Previos

Antes de ejecutar la aplicación, asegúrate de contar con los siguientes componentes instalados:

1. **Python 3.10 o superior**
2. **LM Studio** (para ejecutar modelos locales)
3. **Modelo Qwen3-1.7B** 
4. Dependencias del proyecto incluidas en `requirements.txt`

---

# 🚀 Instalación

## Paso 1: Instalar dependencias

```bash
pip install -r requirements.txt
```

## Paso 2: Entrenar el modelo ML (Primera vez)

Ejecutar el script `modelo.py` para:
- Cargar y procesar el dataset
- Entrenar el modelo MultiOutputClassifier
- Generar los archivos:
  - `modelo.pkl`
  - `vectorizador.pkl`
  - `mlb.pkl`

```bash
python modelo.py
```

## Paso 3: Iniciar LM Studio

1. Abrir LM Studio
2. Cargar el modelo `Qwen3-1.7B` 
3. Activar el servidor local de inferencia en:

```text
http://localhost:1234/v1
```

## Paso 4: Ejecutar la aplicación

Ejecutar el script `app.py` para iniciar la interfaz:

```bash
python app.py
```

La interfaz estará disponible en:

```text
http://localhost:7860
```

---

# 📚 Funcionalidades Principales

## Validación Estructural

El sistema analiza reportes PDF y verifica:

* ✔ **Detección de secciones** - Identifica automáticamente las secciones del reporte
* ✔ **Validación de completitud** - Detecta secciones faltantes
* ✔ **Análisis de extensión** - Cuenta palabras por sección
* ✔ **Validación de mínimos** - Verifica que cada sección cumpla con el mínimo de palabras requerido

### Secciones Detectadas

El sistema reconoce las siguientes secciones:

1. **Introducción** - Mínimo 150 palabras
2. **Planteamiento del Problema** - Mínimo 150 palabras
3. **Objetivos** - Mínimo 80 palabras
4. **Marco Teórico** - Mínimo 500 palabras
5. **Desarrollo/Metodología** - Mínimo 800 palabras
6. **Resultados** - Mínimo 300 palabras
7. **Conclusiones** - Mínimo 150 palabras
8. **Referencias** - Mínimo 20 palabras

---

## Diagnóstico Inteligente

### Machine Learning (ML)

El sistema utiliza un modelo RandomForest multiclase para identificar problemas específicos en el reporte:

- Problemas de estructura
- Deficiencias de contenido
- Inconsistencias en la redacción
- Falta de claridad

### IA Generativa Local (Qwen3)

Proporciona observaciones detalladas sobre:

- Calidad de la estructura general
- Claridad del contenido
- Coherencia y fluidez
- Recomendaciones de mejora

---

## Estados del Reporte

El sistema clasifica el estado general del reporte en:

* **Correcto** - Cumple con todas las secciones (100% de cumplimiento)
* **Incompleto** - Falta alguna sección (60-99% de cumplimiento)
* **Deficiente** - Faltan varias secciones (30-59% de cumplimiento)
* **Mal estructurado** - Faltan la mayoría de secciones (<30% de cumplimiento)

---

# 📖 Uso de la Aplicación

## Validar un Reporte

1. Abrir la interfaz en `http://localhost:7860`
2. Hacer clic en "Subir PDF"
3. Seleccionar el archivo PDF del reporte
4. El sistema procesará automáticamente el documento
5. Revisar el resultado que incluye:
   - Estado general
   - Porcentaje de cumplimiento estructural
   - Secciones detectadas y faltantes
   - Conteo de palabras por sección
   - Diagnóstico IA (Machine Learning)
   - Observaciones de IA Local (Qwen3)

---

# 🏗️ Arquitectura General

```
Reporte PDF
    │
    ▼
Extracción de Texto
    │
    ▼
Normalización de Texto
    │
    ▼
Detección de Secciones
    │
    ├─────────────────────┬──────────────────────┐
    │                     │                      │
    ▼                     ▼                      ▼
Análisis Estructura   Conteo Palabras    Machine Learning
    │                     │                      │
    │                     │              Modelo RandomForest
    │                     │              + Vectorizador TF-IDF
    │                     │                      │
    └─────────────────────┼──────────────────────┘
                          │
                          ▼
                LM Studio (Qwen3 Local)
                          │
                          ▼
                    Interfaz Gradio
                          │
                          ▼
                    Resultado Completo
```

---

# 🔒 Privacidad y Seguridad

Todas las operaciones se realizan localmente:

* ✓ No se envían documentos a servicios externos
* ✓ No se utilizan APIs en la nube
* ✓ Los datos permanecen completamente en el equipo del usuario
* ✓ Compatible con entornos sin conexión a Internet
* ✓ Modelos de IA ejecutados localmente

---

# ⚙️ Configuración Personalizada

Puedes personalizar los requisitos mínimos de palabras editando los valores en `app.py`:

```python
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
```

También puedes agregar nuevas variantes de secciones en el diccionario `SECCIONES`.

---

# 🔧 Solución de Problemas

### Error: "No se pudo conectar a LM Studio"

- Verifica que LM Studio esté abierto y corriendo
- Asegúrate de que el servidor local esté activado en `localhost:1234`
- Reinicia LM Studio si es necesario

### Error: "No se encontraron los archivos .pkl"

- Ejecuta primero el script `modelo.py` para entrenar y generar los modelos
- Verifica que los archivos estén en la misma carpeta que `app.py`

### Error: "Modelo de IA no disponible"

- Verifica que tengas el modelo Qwen3-1.7B descargado en LM Studio
- Comprueba el nombre exacto del modelo en LM Studio

---

# 📊 Interpretación de Resultados

### Porcentaje de Cumplimiento Estructural

Indica qué porcentaje de las secciones esperadas están presentes en el reporte.

### Conteo de Palabras

Muestra la cantidad de palabras en cada sección y si cumple con el mínimo requerido:
- **Correcto**: Supera el mínimo requerido
- **Muy corto**: No alcanza el mínimo

### Diagnóstico IA (ML)

Lista los problemas específicos detectados por el modelo de Machine Learning, como:
- Falta de estructura clara
- Contenido insuficiente
- Secciones mal distribuidas

