import pyttsx3
import threading
import queue
import traceback


class AudioAnnouncer:
    """
    Runs a persistent background thread that speaks queued messages.

    IMPORTANT Windows/pyttsx3 quirk: reusing ONE engine object across
    multiple say()+runAndWait() cycles is unreliable on Windows -- the
    SAPI5 driver's internal state gets stuck after the first successful
    call, and every announcement after that silently does nothing (no
    exception, nothing printed -- it just never speaks). This is a
    well-documented pyttsx3 issue, not a bug in this code.

    The fix: build a FRESH pyttsx3 engine for every single message
    instead of reusing one across the whole thread's lifetime. Slightly
    slower per call (engine init takes ~100-300ms) but reliably speaks
    every message instead of only the first one.
    """

    def __init__(self, rate=160):
        self.q = queue.Queue()
        self.rate = rate
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        # REQUIRED FOR WINDOWS: only needs to be called once for this thread.
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except ImportError:
            pass

        while True:
            msg = self.q.get()
            try:
                # fresh engine per message -- avoids the "only speaks once" bug
                engine = pyttsx3.init()
                engine.setProperty("rate", self.rate)
                engine.say(msg)
                engine.runAndWait()
                engine.stop()
                del engine
            except Exception:
                print("AudioAnnouncer: failed to speak a message — see traceback below.")
                traceback.print_exc()
            finally:
                self.q.task_done()

    def announce(self, message):
        self.q.put(message)