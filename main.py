import io

import numpy as np
import streamlit as st
import keras
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image
from streamlit_tensorboard import st_tensorboard

import constant


def main():
    # Load trained model
    classifier = load_model('Trained_model.h5')

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


    st.subheader("TENSORBOARD")
    st_tensorboard(logdir=constant.LOG_DIR, port=6006, width=1080)


if __name__ == "__main__":
    main()
