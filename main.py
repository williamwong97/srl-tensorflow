import streamlit as st
from keras.preprocessing import image
import numpy as np
from keras.models import load_model
from PIL import Image
import io
import pandas as pd
from streamlit_tensorboard import st_tensorboard

from cnn_model import log_dir

# Load trained model
classifier = load_model('Trained_model.h5')
log_data = pd.read_csv('', sep=',', engine='python')
# Start TensorBoard
st_tensorboard(logdir=log_dir(), port=6006, width=1080)

# summarize history for accuracy


# Start UI
st.title("Sign language detector - Project AI")

uploaded_file = st.file_uploader("Pick a file")

# Load file and start predict
if uploaded_file is not None:
    # Convert the file to an opencv image.
    bytes_data = uploaded_file.getvalue()

    # Now do something with the image! For example, let's display it:
    st.image(bytes_data, caption="Input sign language image")

    img = Image.open(io.BytesIO(bytes_data))
    img = img.convert('RGB')
    img = img.resize((64, 64), Image.NEAREST)
    img = image.img_to_array(img)
    test_image = np.expand_dims(img, axis=0)

    result = classifier.predict(test_image)
    if result[0][0] == 1:
        st.write('A')
    elif result[0][1] == 1:
        st.write('B')
    elif result[0][2] == 1:
        st.write('C')
    elif result[0][3] == 1:
        st.write('D')
    elif result[0][4] == 1:
        st.write('E')
    elif result[0][5] == 1:
        st.write('F')
    elif result[0][6] == 1:
        st.write('G')
    elif result[0][7] == 1:
        st.write('H')
    elif result[0][8] == 1:
        st.write('I')
    elif result[0][9] == 1:
        st.write('J')
    elif result[0][10] == 1:
        st.write('K')
    elif result[0][11] == 1:
        st.write('L')
    elif result[0][12] == 1:
        st.write('M')
    elif result[0][13] == 1:
        st.write('N')
    elif result[0][14] == 1:
        st.write('O')
    elif result[0][15] == 1:
        st.write('P')
    elif result[0][16] == 1:
        st.write('Q')
    elif result[0][17] == 1:
        st.write('R')
    elif result[0][18] == 1:
        st.write('S')
    elif result[0][19] == 1:
        st.write('T')
    elif result[0][20] == 1:
        st.write('U')
    elif result[0][21] == 1:
        st.write('V')
    elif result[0][22] == 1:
        st.write('W')
    elif result[0][23] == 1:
        st.write('X')
    elif result[0][24] == 1:
        st.write('Y')
    elif result[0][25] == 1:
        st.write('Z')
