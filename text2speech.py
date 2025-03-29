import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from pygame import mixer
import torch
import torchaudio
import matplotlib.pyplot as plt

mixer.init()
torch.random.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
processor = bundle.get_text_processor()
tacotron2 = bundle.get_tacotron2().to(device)
vocoder = bundle.get_vocoder().to(device)

class App:
    def __init__(self, master):
        self.master = master
        master.title('Text to Speech')
        
        self.text_entry = tk.Entry(master)
        self.text_entry.pack()
        
        self.generate_button = tk.Button(master, text='Generate', command=self.generate)
        self.generate_button.pack()
        
        self.figure = Figure(figsize=(5, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=master)
        self.canvas.get_tk_widget().pack()
        self.canvas.mpl_connect("button_press_event", self.play_audio)
    
    def generate(self):
        text = self.text_entry.get()
        with torch.inference_mode():
            processed, lengths = processor(text)
            processed = processed.to(device)
            lengths = lengths.to(device)
            spec, spec_lengths, *_ = tacotron2.infer(processed, lengths)
            waveforms, lengths = vocoder(spec, spec_lengths)
        
        self.figure.clear()
        a = self.figure.add_subplot(211)
        a.imshow(spec[0].cpu().detach(), origin="lower", aspect="auto")
        b = self.figure.add_subplot(212)
        b.plot(waveforms[0].cpu().detach())
        self.canvas.draw()
        
        torchaudio.save("output.wav", waveforms, sample_rate=vocoder.sample_rate)
    
    def play_audio(self, event):
        try:
            mixer.music.load('output.wav')
            mixer.music.play()
        except Exception as e:
            messagebox.showerror("Error", str(e))

root = tk.Tk()
app = App(root)
root.mainloop()