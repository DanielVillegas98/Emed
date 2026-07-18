from fpdf import FPDF
import datetime
import speech_recognition as sr
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
from google import genai
from google.genai import types
from gtts import gTTS
import pygame
import time

# Initialize Gemini Client
client = genai.Client()
pygame.mixer.init()

def speak(text):
    print(f"\nNeuroEdge (AI): {text}")
    try:
        # Uses Google's Cloud TTS for a perfectly natural English voice
        tts = gTTS(text=text, lang='en', tld='us')
        filename = "response.mp3"
        tts.save(filename)
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        pygame.mixer.music.unload()
        if os.path.exists(filename):
            os.remove(filename)
    except Exception as e:
        print(f"[Audio Error: {e}]")

def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n[Listening to Grandpa...]")
        recognizer.adjust_for_ambient_noise(source, duration=1) 
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            text = recognizer.recognize_google(audio)
            print(f"Grandpa: {text}")
            return text
        except sr.WaitTimeoutError:
            print("[No speech detected]")
            return None
        except sr.UnknownValueError:
            print("[Speech unintelligible]")
            return "unintelligible"
        except Exception as e:
            print(f"[Microphone Error: {e}]")
            return None

def run_voice_triage_loop():
    # We are making the rules extremely strict to prevent weird sentences
    system_prompt = """
    You are an ambient health assistant for an elderly Parkinson's patient named Arthur. 
    You just detected that he fell. 
    Your goal is to triage the situation. 
    STRICT RULES:
    1. Ask ONE clear, complete sentence at a time.
    2. Never use incomplete sentences or trailing words.
    3. If he says he is in pain, his back hurts, or he is not safe, reply with EXACTLY: "I am calling an ambulance right now, please stay still. [END_TRIAGE_EMERGENCY]"
    4. If he is fine and safe, reply with EXACTLY: "Take your time getting up. I will log this for your doctor. [END_TRIAGE_SAFE]"
    """

    gemini_history = []
    clinical_history = []

    initial_message = "Arthur, I noticed you might have fallen. Are you okay? Where does it hurt?"
    speak(initial_message)
    
    gemini_history.append(
        types.Content(role="model", parts=[types.Part.from_text(text=initial_message)])
    )
    clinical_history.append({"role": "AI", "content": initial_message})

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.1, # Lowered to 0.1 to force strict, stable, complete sentences
    )

    while True:
        patient_response = listen()
        
        if not patient_response:
            speak("Arthur, I didn't catch that. Are you in pain?")
            continue
            
        gemini_history.append(
            types.Content(role="user", parts=[types.Part.from_text(text=patient_response)])
        )
        clinical_history.append({"role": "Patient", "content": patient_response})
        
        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash", 
                contents=gemini_history,
                config=config
            )
            
            ai_reply = response.text
            
            if not ai_reply:
                speak("I am having trouble processing that. Can you tell me if you are safe?")
                continue
            
            if "[END_TRIAGE_EMERGENCY]" in ai_reply:
                clean_reply = ai_reply.replace("[END_TRIAGE_EMERGENCY]", "").strip()
                speak(clean_reply)
                clinical_history.append({"role": "AI", "content": clean_reply})
                generate_clinical_summary(clinical_history, outcome="Emergency Services Dispatched")
                break
                
            elif "[END_TRIAGE_SAFE]" in ai_reply:
                clean_reply = ai_reply.replace("[END_TRIAGE_SAFE]", "").strip()
                speak(clean_reply)
                clinical_history.append({"role": "AI", "content": clean_reply})
                generate_clinical_summary(clinical_history, outcome="Patient recovered independently")
                break
                
            else:
                speak(ai_reply)
                gemini_history.append(
                    types.Content(role="model", parts=[types.Part.from_text(text=ai_reply)])
                )
                clinical_history.append({"role": "AI", "content": ai_reply})
                
        except Exception as e:
            print(f"\nAPI Error: {e}")
            speak("I am having trouble connecting to the network, but I am alerting your family now.")
            # THIS IS THE CRITICAL FIX: Generate the PDF even if the system crashes!
            generate_clinical_summary(clinical_history, outcome="System Fallback - Network/API Limit Reached")
            break


def generate_clinical_summary(history, outcome):
    print("\n" + "="*50)
    print("📋 GENERATING CLINICAL PDF SUMMARY FOR NEUROLOGIST...")
    print("="*50)
    
    # Inicializar PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Encabezado del Documento
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt="NeuroEdge: Ambient Care Clinical Report", ln=True, align='C')
    pdf.ln(5)
    
    # Datos del Paciente (Simulados para el Demo)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, txt="Patient: Arthur | Age: 78 | Condition: Parkinson's Disease", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 8, txt=f"Report Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(0, 8, txt="Reviewing Physician: Dr. Sarah Jenkins, Neurology", ln=True)
    current_y = pdf.get_y()
    pdf.line(10, current_y + 2, 200, current_y + 2)
    pdf.ln(10)
    
    # Sección 1: Alerta del Incidente Principal (Datos Reales del Demo)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt="1. Acute Incident Alert: Fall Detected", ln=True)
    
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(35, 8, txt="AI Assessment: ")
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 8, txt=f"{outcome}", ln=True)
    pdf.ln(2)
    
    # Transcripción del Triage
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 8, txt="Automated Voice Triage Transcript:", ln=True)
    pdf.set_font("Arial", 'I', 10)
    
    for msg in history:
        speaker = msg['role'].upper()
        clean_text = msg['content'].replace('[END_TRIAGE_EMERGENCY]', '').replace('[END_TRIAGE_SAFE]', '')
        # Usamos multi_cell para que el texto largo no se salga de la página
        pdf.multi_cell(0, 6, txt=f"{speaker}: {clean_text}")
    
    pdf.ln(10)
    
    # Sección 2: Contexto Longitudinal (Datos Simulados según nuestro Pitch)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt="2. 30-Day Longitudinal Biomarkers (Ambient Tracking)", ln=True)
    
    pdf.set_font("Arial", '', 11)
    data_points = [
        chr(149) + " Medication Adherence: 94% overall. Morning Levodopa doses delayed by >30 mins on 4 occasions.",
        chr(149) + " Voice Biomarkers: Progressive hypophonia detected. Average vocal amplitude decreased by 12% over 3 weeks.",
        chr(149) + " Motor Function (Gait): 15% increase in freezing of gait episodes, primarily between 07:00 and 09:00.",
        chr(149) + " Tremor Analysis: Amplitude increased by 8% in the left upper extremity during evening periods.",
        chr(149) + " Sleep Quality: Frequent nocturnal awakenings detected. Potential REM sleep behavior disorder indicators."
    ]
    
    for point in data_points:
        pdf.multi_cell(0, 6, txt=point)
        pdf.ln(2)
    
    pdf.ln(5)
    
    # Sección 3: Recomendación del Sistema
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(220, 53, 69) # Letra roja para alertas
    pdf.cell(0, 8, txt="SYSTEM RECOMMENDATION:", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.set_text_color(0, 0, 0) # Regresar a negro
    pdf.multi_cell(0, 6, txt="Based on the recent fall and the longitudinal increase in freezing of gait, an early medication titration review is highly recommended before the next scheduled 6-month clinic visit.")
    
    # Guardar el archivo
    filename = "Arthur_Clinical_Report.pdf"
    pdf.output(filename)
    
    print(f"✅ PDF successfully generated and saved directly in your folder as: {filename}")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_voice_triage_loop()

