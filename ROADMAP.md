# Roadmap

### 26/01/17 - Michele Madaschi

creato il repository, inizializzato con il codice di py_eyetracker_v1.0
per testare lo script:

    cd py_eyetracker_v1.0
    python main.py ../test_images/picard.jpg

Nota 1: lo script funziona solo per le immagini in cui sono visibili entrambi gli occhi  
Nota 2: per funzionare, necessita dei moduli:

    skimage
    matplotlib
    numpy
    cv2

### 13/51/17 - Michele Madaschi

modificato l'algoritmo per il rilevamento delle pupille:

* miglior pre-processing dell'immagine
* thresholding e divisione in regioni

aggiunto rilevamento degli angoli dell'occhio, usando Harris corner detection ed
un euristica geometrica basata sulla posizione della pupilla

cleanup del codice
