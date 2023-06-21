from music21 import *
from tkinter import filedialog
import tkinter as tk
import random
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
import torch.nn.functional as F
import torch
import numpy as np
import cv2
from music21 import *
environment.set(
    'musicxmlPath', '/Applications/MuseScore 4.app/Contents/MacOS/mscore')


# catergorize the 80 items that could be detected
person_array = np.array([0])
vehicle_array = np.array([1, 2, 3, 4, 5, 6, 7, 8])
road_object_array = np.array([9, 10, 11, 12, 13])
home_animal_array = np.array([14, 15, 16, 18])
wild_animal_array = np.array([17, 19, 20, 21, 22, 23])
bags_array = np.array([24, 25, 26, 27, 28])
sports_related_array = np.array([29, 30, 31, 32, 33, 34, 35, 36, 37, 38])
food_related_object_array = np.array([39, 40, 41, 42, 43, 44, 45])
food_array = np.array([46, 47, 48, 49, 50, 51, 52, 53, 54, 55])
home_object_array = np.array(
    [56, 57, 58, 59, 60, 61, 62, 71, 73, 75, 76, 77, 79])
electronic_array = np.array([63, 64, 65, 66, 67, 68, 69, 70, 72, 74, 78])

# list of emotions that it could output
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


# each major scale
# happy
C_Major = ["C", "D", "E", "F", "G", "A", "B", "C5"]
# Suprise
D_Major = ["D", "E", "F#", "G", "A", "B", "C#", "D5"]
# fear
G_Major = ["G", "A", "B", "C", "D", "E", "F#", "G5"]
# happy
A_Major = ["A", "B", "C#5", "D5", "E5", "F#5", "G#5", "A5"]
# angry
E_Major = ["E", "F#", "G#", "A", "B", "C#5", "D#5", "E5"]
# angry
F_Major = ["F", "G", "A", "A#", "C5", "D5", "E5", "F5"]

# each minor scale
# neutral
A_Minor = ["A", "B", "C5", "D5", "E5", "F5", "G5", "A5"]
# fear
E_Minor = ["E", "F#", "G", "A", "B", "C5", "D5", "E5"]
# sad
C_Minor = ["C", "D", "Eb", "F", "G", "Ab", "Bb", "C5"]
# neutral
D_Minor = ["D", "E", "F", "G", "A", "Bb", "C", "D5"]
# fear
G_Minor = ["G", "A", "Bb", "C5", "D5", "Eb5", "F5", "G5"]
# sad
B_Minor = ["B", "C#5", "D5", "E5", "F#5", "G5", "A5", "B5"]

# List of major chords
c_major_chord = ['C', 'E', 'G']
a_major_chord = ['A', 'C#', 'E']
g_major_chord = ['G', 'B', 'D']
f_major_chord = ['F', 'A', 'C']
d_minor_chord = ['D', 'F', 'A']
d_major_chord = ['D', 'F#', 'A']
bb_major_chord = ['Bb', 'D', 'F']
e_major_chord = ['E', 'G#', 'B']
a_minor_chord = ['A', 'C', 'E']
c_minor_chord = ['C', 'Eb', 'G']


# Create a melody using a Markov chain
melody = stream.Part()

# vggface2 for emotion recognition
model = InceptionResnetV1(pretrained='vggface2').eval()
# model for recongizing object from the photo
secound_model = YOLO("yolov8x.pt")


# create the main window of the GUI
root = tk.Tk()

# hide the main window as we won't use it
root.withdraw()

# prompt the user to select an image file using a file dialog
file_path = filedialog.askopenfilename()

# open the image file using the CV2 library
image = cv2.imread(file_path)


# resize for vggface2
img = cv2.resize(image, (160, 160))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_tensor = transforms.ToTensor()(img)

# Generate predictions and extract emotion labels
with torch.no_grad():
    img_tensor = img_tensor.unsqueeze(0)
    preds = model(img_tensor)
    probs = F.softmax(preds, dim=1)

emotion_probs = dict(zip(emotions, probs.tolist()[0]))
emotion = max(emotion_probs, key=emotion_probs.get)
print('Predicted emotion:', emotion)

# map emotion into which key and tempo use
if emotion == "happy":
    tempos = random.randint(120, 140)

    choice_of_second_instrument = random.randint(1, 2)
    if choice_of_second_instrument == 1:
        instruments = instrument.Vibraphone()
    elif choice_of_second_instrument == 2:
        instruments = instrument.AcousticGuitar()

    selection = random.randint(1, 2)
    if selection == 1:
        choice_of_key = C_Major
        melody.append(key.Key('C'))
    elif selection == 2:
        choice_of_key = A_Major
        melody.append(key.Key('A'))

    positive = 1


elif emotion == "sad":
    positive = 0
    selection = random.randint(1, 2)
    tempos = random.randint(50, 70)
    if selection == 1:
        choice_of_key = C_Minor
        melody.append(key.Key('C#'))
    elif selection == 2:
        choice_of_key = B_Minor
        melody.append(key.Key('B#'))

    choice_of_second_instrument = random.randint(1, 2)
    if choice_of_second_instrument == 1:
        instruments = instrument.Bass()
    elif choice_of_second_instrument == 2:
        instruments = instrument.Oboe()


elif emotion == "fear" or emotion == "disgust":
    positive = 0
    selection = random.randint(1, 2, 3)
    tempos = random.randint(60, 80)
    instruments = instrument.Bass()
    if selection == 1:
        choice_of_key = G_Major
        melody.append(key.Key('G'))
    elif selection == 2:
        choice_of_key = G_Minor
        melody.append(key.Key('G#'))
    elif selection == 3:
        choice_of_key = E_Minor
        melody.append(key.Key('E#'))

    choice_of_second_instrument = random.randint(1, 2)
    if choice_of_second_instrument == 1:
        instruments = instrument.Bass()
    elif choice_of_second_instrument == 2:
        instruments = instrument.Contrabass()


elif emotion == "neutral" or "no detections":
    positive = 2
    selection = random.randint(1, 2)
    tempos = random.randint(90, 110)
    if selection == 1:
        choice_of_key = A_Minor
        melody.append(key.Key('A#'))
    elif selection == 2:
        choice_of_key = D_Minor
        melody.append(key.Key('D#'))

    choice_of_second_instrument = random.randint(1, 2)
    if choice_of_second_instrument == 1:
        instruments = instrument.AcousticBass()
    elif choice_of_second_instrument == 2:
        instruments = instrument.Marimba()


elif emotion == "suprise":
    positive = 2
    tempos = random.randint(150, 180)
    choice_of_key = D_Major
    melody.append(key.Key('D'))

    choice_of_second_instrument = random.randint(1, 2)
    if choice_of_second_instrument == 1:
        instruments = instrument.Harp()
    elif choice_of_second_instrument == 2:
        instruments = instrument.Xylophone()


elif emotion == "angry":
    positive = 1
    selection = random.randint(1, 2)
    tempos = random.randint(140, 180)
    if selection == 1:
        choice_of_key = E_Major
        melody.append(key.Key('E'))
    elif selection == 2:
        choice_of_key = F_Major
        melody.append(key.Key('F'))

    choice_of_second_instrument = random.randint(1, 2)
    if choice_of_second_instrument == 1:
        instruments = instrument.SteelDrum()
    elif choice_of_second_instrument == 2:
        instruments = instrument.ElectricGuitar()

# generate the melody

note_length = ['whole', 'half', 'quarter', 'eighth', '16th']
slow_note_length = ['whole', 'half', 'quarter']
quick_note_length = ['quarter', 'eighth', '16th']

# Get detection results
results = secound_model(image)
result = results[0]

# the number code of extracted object
obtained_array = np.array(result.boxes.cls.cpu(), dtype="int")

for i in range(16):

    current_note = random.choice(choice_of_key)

    curate = random.randint(0, 4)
    curated = random.randint(0,2)
    numberNeed = random.randint(1, 10)
    
    if positive == 2:
        yes = duration.Duration(note_length[curate])
    
    elif positive == 1:
        if numberNeed >= 6:
            yes = duration.Duration(note_length[curate]) 
        else:
            yes = duration.Duration(quick_note_length[curated]) 
            
    elif positive == 0:
        if numberNeed >= 6:
            yes = duration.Duration(note_length[curate]) 
        else:
            yes = duration.Duration(slow_note_length[curated]) 
            
    for obtained in obtained_array:

        if np.isin(obtained, person_array):
            if current_note == 'C':
                if positive == 1:
                    melody.append(chord.Chord(
                        c_major_chord, duration=yes))
                else:
                    melody.append(chord.Chord(
                        c_minor_chord, duration=yes))
            else:
                melody.append(note.Note(current_note, duration=yes))

        elif np.isin(obtained, vehicle_array):
            if current_note == 'A':
                if positive == 1:
                    melody.append(chord.Chord(
                        a_major_chord, duration=yes))
                else:
                    melody.append(chord.Chord(
                        a_minor_chord, duration=yes))
            else:
                melody.append(note.Note(current_note, duration=yes))

        elif np.isin(obtained, road_object_array):
            if current_note == 'G':
                melody.append(chord.Chord(g_major_chord, duration=yes))
            else:
                melody.append(note.Note(current_note, duration=yes))

        elif np.isin(obtained, home_animal_array):
            if current_note == 'F':
                melody.append(chord.Chord(f_major_chord, duration=yes))
            else:
                melody.append(note.Note(current_note, duration=yes))

        elif np.isin(obtained, wild_animal_array):
            if current_note == 'D':
                if positive == 1:
                    melody.append(chord.Chord(
                        d_major_chord, duration=yes))
                else:
                    melody.append(chord.Chord(
                        d_minor_chord, duration=yes))
            else:
                melody.append(note.Note(current_note, duration=yes))

        elif np.isin(obtained, bags_array):
            if current_note == 'Bb' or current_note == 'A#':
                melody.append(chord.Chord(bb_major_chord, duration=yes))
            else:
                melody.append(note.Note(current_note, duration=yes))

        elif np.isin(obtained, sports_related_array):
            if current_note == 'E':
                melody.append(chord.Chord(e_major_chord, duration=yes))
            else:
                melody.append(note.Note(current_note, duration=yes))

        elif np.isin(obtained, food_related_object_array):
            if current_note == 'A':
                if positive == 1:
                    melody.append(chord.Chord(
                        a_major_chord, duration=yes))
                else:
                    melody.append(chord.Chord(
                        a_minor_chord, duration=yes))
            else:
                melody.append(note.Note(current_note, duration=yes))

        elif np.isin(obtained, food_array):
            if current_note == 'G':
                melody.append(chord.Chord(g_major_chord, duration=yes))
            else:
                melody.append(note.Note(current_note, duration=yes))

        elif np.isin(obtained, home_object_array):
            if current_note == 'C':
                if positive == 1:
                    melody.append(chord.Chord(c_major_chord, duration=yes))
                else:
                    melody.append(chord.Chord(c_minor_chord, duration=yes))
            else:
                melody.append(note.Note(current_note, duration=yes))

        elif np.isin(obtained, electronic_array):
            if current_note == 'D':
                if positive == 1:
                    melody.append(chord.Chord(d_major_chord, duration=yes))
                else:
                    melody.append(chord.Chord(d_minor_chord, duration=yes))
            else:
                melody.append(note.Note(current_note, duration=yes))
                
    if obtained_array.size == 0:
        melody.append(note.Note(current_note, duration=yes))

tempo_mark = tempo.MetronomeMark(number=tempos)
melody.insert(0, tempo_mark)


melody.show()


# create a drum beat with a bass drum on the first beat and a snare drum on the third beat

drum_beat = stream.Part()
drum_beat.append(instruments)

current_drum = random.choice(choice_of_key)
for i in range(16):
    Drandom_float = random.uniform(0, 1)
    Dround_float = round(Drandom_float, 1)

    if Dround_float > 0.7:
        drum_beat.append(note.Note(current_drum))
    else:
        drum_beat.append(note.Rest(quarterLength=1.0))

    next_drum = random.choice(choice_of_key)
    current_drum = next_drum


pitch_boost = 0



# Combine the two streams
combined_stream = stream.Stream()
combined_stream.insert(0, melody)
combined_stream.insert(0, drum_beat)

# Show the score
combined_stream.show()