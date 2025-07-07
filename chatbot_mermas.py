import sqlite3
import json
import time
# --- NUEVA IMPORTACIÓN ---
import google.generativeai as genai

# --- CONFIGURACIÓN ---
DB_FILE = 'datamart_mermas.db'
# --- PEGA AQUÍ TU CLAVE DE API DE GOOGLE GEMINI ---
GOOGLE_API_KEY = "AIzaSyDwRgy2TCCJr7j4WxeNJw_zyZNnWV6o7yA" # <-- ¡REEMPLAZA ESTO CON TU CLAVE REAL!

# Configura la API de Google
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    # Selecciona el modelo que vamos a usar
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    print(f"Error al configurar la API de Google. Asegúrate de que la clave es correcta. Error: {e}")
    exit()

# Estructura del datamart (sin cambios)
ESTRUCTURA_DATAMART = """
-- Tabla de Hechos
TABLA hechos_mermas
merma_unidad (real), merma_monto (integer), id_producto (integer), id_ubicacion (integer), id_tiempo (integer), id_motivo (integer)

-- Dimensiones
TABLA dim_producto
id_producto (integer), descripcion (text), categoria (text), seccion (text), linea (text)

TABLA dim_ubicacion
id_ubicacion (integer), tienda (text), comuna (text), region (text)

TABLA dim_motivo
id_motivo (integer), motivo (text), ubicacion_motivo (text)

TABLA dim_tiempo
id_tiempo (integer), fecha (date), dia (integer), nombre_dia (text), semana (integer), mes (integer), nombre_me (text), trimestre (integer), año (integer), semestre (integer)
"""

def obtener_consulta_sql(pregunta):
    """Genera una consulta SQL usando la API de Google Gemini."""
    # El prompt es muy similar, Gemini lo entiende perfectamente.
    prompt = f"""
Tu tarea es generar una única consulta SQL para una base de datos SQLite a partir de una pregunta de usuario.
La base de datos tiene la siguiente estructura:
{ESTRUCTURA_DATAMART}

Reglas:
- La consulta debe unir (JOIN) la tabla `hechos_mermas` con las dimensiones necesarias.
- Usa `LOWER()` y `LIKE '%...%'` para búsquedas de texto flexibles.
- Usa `SUM()` o `COUNT()` para agregaciones, junto con `GROUP BY`.
- Usa `ORDER BY ... DESC LIMIT ...` para rankings.

Pregunta del usuario: "{pregunta}"

Responde únicamente con la consulta SQL. No incluyas explicaciones, ni la palabra "sql", ni los marcadores de código ```.
"""
    try:
        # Hacemos la llamada a la API de Gemini
        response = model.generate_content(prompt)
        # Extraemos y limpiamos la consulta SQL
        sql_query = response.text.strip()
        return sql_query
    except Exception as e:
        print(f"Error durante la generación de SQL con la API de Gemini: {e}")
        return None

# Las funciones ejecutar_sql, generar_respuesta_final y main pueden permanecer igual.
# Las incluyo aquí por completitud.

def ejecutar_sql(sql):
    """Ejecuta una consulta SQL en la base de datos SQLite."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(sql)
    results = [dict(row) for row in cur.fetchall()]
    cur.close()
    conn.close()
    return results

def generar_respuesta_final(resultado_sql, pregunta):
    """Genera una respuesta en lenguaje natural basada en reglas."""
    if not resultado_sql:
        return "No se encontró información para esa solicitud."

    respuesta = f"Resultados para: '{pregunta}'\n\n"
    try:
        headers = resultado_sql[0].keys()
        col_widths = {h: max(len(h), max(len(str(row[h])) for row in resultado_sql) if resultado_sql else 0) for h in headers}
        
        header_line = " | ".join(f"{h.upper():<{col_widths[h]}}" for h in headers)
        respuesta += header_line + "\n"
        respuesta += "-" * len(header_line) + "\n"
        
        for row in resultado_sql:
            row_items = []
            for key, value in row.items():
                if 'monto' in str(key).lower() and isinstance(value, (int, float)):
                    formatted_val = f"${value:,.0f}".replace(",", ".")
                    row_items.append(f"{formatted_val:<{col_widths[key]}}")
                else:
                    row_items.append(f"{str(value):<{col_widths[key]}}")
            respuesta += " | ".join(row_items) + "\n"
    except (IndexError, TypeError):
        # Si hay un problema con el formato (ej. no hay resultados), muestra la info cruda
        respuesta += json.dumps(resultado_sql, indent=2, ensure_ascii=False)
        
    return respuesta.strip()


def main():
    """Función principal para ejecutar el chatbot."""
    if "YOUR_GOOGLE_API_KEY" in GOOGLE_API_KEY:
        print("¡ERROR! Debes reemplazar 'YOUR_GOOGLE_API_KEY' con tu clave de API real de Google Gemini en el script.")
        return

    print("\nChatbot de Análisis de Mermas (con API Gratuita de Google Gemini) iniciado.")
    
    while True:
        pregunta = input("\nIngrese su pregunta (o escriba 'salir' para finalizar): ")
        if pregunta.lower() == 'salir':
            print("Chat finalizado.")
            break
        
        try:
            print("Generando consulta SQL a través de la API de Google Gemini...")
            sql_query = obtener_consulta_sql(pregunta)
            if not sql_query:
                print("No se pudo generar una consulta SQL.")
                continue
            
            print(f"SQL Generado: {sql_query}")

            sql_resultados = ejecutar_sql(sql_query)
            
            respuesta_final = generar_respuesta_final(sql_resultados, pregunta)
            print(f"\nRespuesta: \n{respuesta_final}")
            
        except sqlite3.Error as e:
            print(f"Error de base de datos: La consulta SQL generada podría ser inválida. Detalles: {e}")
        except Exception as e:
            print(f"Ocurrió un error inesperado: {e}")

if __name__ == "__main__":
    main()