from pathlib import Path

src = Path("/mnt/data/Pasted text(20260902-183255).txt")
text = src.read_text(encoding="utf-8")

old = 'data_mode = st.sidebar.radio("Operation Mode", ["Live Camera", "Simulation"])'
new = '''data_mode = st.sidebar.radio(
    "Operation Mode",
    ["Simulation", "Live Camera"],
    index=0
)'''
if old not in text:
    raise ValueError("Operation Mode line was not found")

text = text.replace(old, new, 1)

old_audio = '''@st.cache_resource
def get_announcer():
    return AudioAnnouncer()

announcer = get_announcer()'''
new_audio = '''@st.cache_resource
def get_announcer():
    try:
        return AudioAnnouncer()
    except Exception as e:
        # Audio/TTS may not be available on Streamlit Cloud.
        print(f"Audio announcer disabled: {e}")
        return None

announcer = get_announcer()'''
if old_audio not in text:
    raise ValueError("Audio initialization block was not found")

text = text.replace(old_audio, new_audio, 1)

old_announce = '''            if should_speak:
                clean_msg = msg.split(" — ")[0].replace("min", "minutes")
                announcer.announce(f"Attention please. {clean_msg}")
                st.session_state.last_audio_time = current_time
                should_speak = False '''
new_announce = '''            if should_speak and announcer is not None:
                clean_msg = msg.split(" — ")[0].replace("min", "minutes")
                try:
                    announcer.announce(f"Attention please. {clean_msg}")
                except Exception as e:
                    print(f"Audio announcement skipped: {e}")
                st.session_state.last_audio_time = current_time
                should_speak = False '''
if old_announce not in text:
    raise ValueError("Audio announce block was not found")

text = text.replace(old_announce, new_announce, 1)

out = Path("/mnt/data/App_cloud_ready.py")
out.write_text(text, encoding="utf-8")

print(f"Created: {out}")
print("Changes made:")
print("1. Simulation is now the default mode.")
print("2. AudioAnnouncer initialization is protected so cloud audio failures won't crash the app.")
print("3. Audio announce calls are protected with a try/except.")
