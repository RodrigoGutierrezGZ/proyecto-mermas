import google.generativeai as genai

# --- PEGA AQUÍ TU CLAVE DE API DE GOOGLE GEMINI ---
GOOGLE_API_KEY = "AIzaSyDwRgy2TCCJr7j4WxeNJw_zyZNnWV6o7yA" # <-- ¡REEMPLAZA ESTO CON TU CLAVE REAL!

genai.configure(api_key=GOOGLE_API_KEY)

print("Buscando modelos disponibles para tu clave de API...")
print("-" * 30)

# Este bucle revisa todos los modelos y nos dice cuáles podemos usar para generar texto
for model in genai.list_models():
  # Comprobamos si el modelo soporta el método 'generateContent'
  if 'generateContent' in model.supported_generation_methods:
    print(f"Modelo compatible encontrado: {model.name}")

print("-" * 30)