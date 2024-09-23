import groq
import numpy
import subprocess
import time


def say(text):
    subprocess.run(["say", text])


class SilenceDetector:
    def __init__(self, threshold=0.01, duration=2):
        self.threshold = threshold
        self.duration = duration
        self.silence_start = None

    def __is_silent(self, data: numpy.ndarray):
        """Check if audio data is below the silence threshold."""
        return numpy.sqrt(numpy.mean(data**2)) < self.threshold

    def is_silence_start(self, data: numpy.ndarray):
        if self.__is_silent(data):
            if self.silence_start is None:
                self.silence_start = time.time()  # Start timing silence
            elif time.time() - self.silence_start >= self.duration:
                return True
        else:
            self.silence_start = None  # Reset silence timer if sound is detected
        return False


def record() -> str:
    import queue
    import sounddevice
    import scipy.io.wavfile as wavfile
    import tempfile

    # # Sampling frequency
    freq = 44100

    recording = numpy.ndarray(1)
    frame_queue = queue.Queue()

    def callback(indata: numpy.ndarray, frames: int, _time, status):
        frame_queue.put(indata.copy())

    with sounddevice.InputStream(channels=1, samplerate=freq, callback=callback):
        print("Recording... Press Ctrl+C to stop.")
        silence_detector = SilenceDetector()
        try:
            while True:
                indata = frame_queue.get()

                if silence_detector.is_silence_start(indata):
                    print("Silence detected, stopping recording.")
                    break

                recording = numpy.append(recording, indata)

        except KeyboardInterrupt:
            print("Recording stopped.")

    print("Recorded is %f frames" % (len(recording) / freq))

    tmpfile = tempfile.mktemp(".wav")

    # This will convert the NumPy array to an audio
    # file with the given sampling frequency
    wavfile.write(tmpfile, freq, recording)

    return tmpfile


client = groq.Groq()

say("What is your question?")

filename = record()

# Open the audio file
with open(filename, "rb") as file:
    # Create a transcription of the audio file
    transcription = client.audio.transcriptions.create(
        file=(filename, file.read()),  # Required audio file
        model="distil-whisper-large-v3-en",  # Required model to use for transcription
        prompt="Specify context or spelling",  # Optional
        response_format="json",  # Optional
        language="en",  # Optional
        temperature=0.0,  # Optional
    )
    # Print the transcription text
    print("Q:" + transcription.text)

    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "Answer to the following question:"},
            {"role": "user", "content": transcription.text},
        ],
    )

    print("A: " + completion.choices[0].message.content)

    try:
        say(completion.choices[0].message.content)
    except KeyboardInterrupt:
        pass
