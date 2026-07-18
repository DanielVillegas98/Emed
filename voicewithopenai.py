import speech_recognition as sr
import pyttsx3
import os
from openai import OpenAI

# Initialize OpenAI Client (Replace with your actual API key)
# os.environ["OPENAI_API_KEY"] = "sk-your-api-key-here"
#client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client = OpenAI(api_key="")  

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()
# Optional: Change voice speed and type to sound more natural/elderly-friendly
engine.setProperty('rate', 160) 

def speak(text):
    """Converts text to speech and plays it."""
    print(f"\nNeuroEdge (AI): {text}")
    engine.say(text)
    engine.runAndWait()

def listen():
    """Listens to the microphone and converts speech to text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n[Listening to Grandpa...]")
        # Adjust for ambient noise briefly
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
            print(f"[Audio Error: {e}]")
            return None

def run_voice_triage_loop():
    """Main loop for the post-fall conversational triage."""
    
    # The System Prompt is crucial. It tells the LLM how to behave.
    system_prompt = """
    You are an ambient health assistant for an elderly Parkinson's patient named Arthur. 
    You just detected that he fell. 
    Your goal is to triage the situation: 
    1. Ask if he is okay and where it hurts.
    2. Determine if he can get up or if he needs an ambulance.
    3. Keep your responses VERY brief, calm, clear, and empathetic. One question at a time.
    4. If he says he is seriously hurt, say you are calling an ambulance and end with '[END_TRIAGE_EMERGENCY]'.
    5. If he says he is fine and can get up, tell him to take his time and end with '[END_TRIAGE_SAFE]'.
    """

    conversation_history = [
        {"role": "system", "content": system_prompt}
    ]

    # Initial greeting trigger
    initial_message = "Arthur, I noticed you might have fallen. Are you okay? Where does it hurt?"
    speak(initial_message)
    conversation_history.append({"role": "assistant", "content": initial_message})

    while True:
        patient_response = listen()
        
        # If we didn't hear anything, ask again
        if not patient_response:
            speak("Arthur, I didn't catch that. Are you in pain?")
            continue
            
        # Add patient response to history
        conversation_history.append({"role": "user", "content": patient_response})
        
        # Call LLM for the next step
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo", # Use 3.5 for speed during the demo
                messages=conversation_history,
                max_tokens=100,
                temperature=0.3
            )
            ai_reply = response.choices[0].message.content
            
            # Check for our secret end-flags
            if "[END_TRIAGE_EMERGENCY]" in ai_reply:
                clean_reply = ai_reply.replace("[END_TRIAGE_EMERGENCY]", "").strip()
                speak(clean_reply)
                generate_clinical_summary(conversation_history, outcome="Emergency Services Dispatched")
                break
                
            elif "[END_TRIAGE_SAFE]" in ai_reply:
                clean_reply = ai_reply.replace("[END_TRIAGE_SAFE]", "").strip()
                speak(clean_reply)
                generate_clinical_summary(conversation_history, outcome="Patient recovered independently")
                break
                
            else:
                speak(ai_reply)
                conversation_history.append({"role": "assistant", "content": ai_reply})
                
        except Exception as e:
            print(f"LLM Error: {e}")
            speak("I am having trouble connecting to the network, but I am alerting your family now.")
            break

def generate_clinical_summary(history, outcome):
    """Mocks the creation of the doctor's clinical report."""
    print("\n" + "="*50)
    print("📋 GENERATING CLINICAL PDF SUMMARY FOR NEUROLOGIST...")
    print("="*50)
    print(f"Outcome: {outcome}")
    print("Incident Log:")
    for msg in history:
        if msg['role'] != 'system':
            speaker = "AI" if msg['role'] == 'assistant' else "Patient"
            # Clean up flags for the report
            clean_text = msg['content'].replace('[END_TRIAGE_EMERGENCY]', '').replace('[END_TRIAGE_SAFE]', '')
            print(f" - {speaker}: {clean_text}")
    print("="*50 + "\n")

# If you want to test just the voice script by itself:
if __name__ == "__main__":
    run_voice_triage_loop()