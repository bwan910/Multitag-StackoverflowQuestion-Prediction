import streamlit as st
from streamlit import components
from PIL import Image
import requests
import joblib
import pickle
import os

# Load tfidf pickle
loaded_clf = open("pickle_files/tfidf.pkl", "rb")
job = joblib.load(loaded_clf)


# load model function
def load_prediction_models(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), "rb"))
    return loaded_model

# predict function
def predict():
    image = Image.open(requests.get(
        'https://sachinkalsi.github.io/blog/data/images/blog_posts/stackoverflow.png', stream=True).raw)
    st.image(image, caption='Stackoverflow')

    # file uploader
    uploaded_file = st.file_uploader("Choose a file", type=['txt'])
    input_text = st.text_area('Enter question:')

   
    
    def file_features():
        # read file value
        bytes_data = uploaded_file.getvalue()
        # transform file value to matrix
        transform_file_matrix = job.transform([bytes_data])
        predictor_dt = load_prediction_models(
            "pickle_files/model.pkl")
        pred_result = predictor_dt.predict(transform_file_matrix)
        
        transform_tags = load_prediction_models("pickle_files/multilabel.pkl")
        output_result = transform_tags.inverse_transform(pred_result)
        st.text('Recommend Tag is {}'.format(output_result))

       
    

    # for text input functions
    def text_features():
        # convert input to numeric matrix value with the TFIDF
        transform_text_matrix = job.transform([input_text])
        print (transform_text_matrix)
        predictor_dt = load_prediction_models(
            "pickle_files/model.pkl")
        #pred_result = predictor_dt.predict(transform_text_matrix)

        transform_tags = load_prediction_models("pickle_files/multilabel.pkl")
        #output_result = transform_tags.inverse_transform(predictor_dt.predict(transform_text_matrix))
        st.text('Recommend Tag is {}'.format(transform_tags.inverse_transform(predictor_dt.predict(transform_text_matrix))))
        print(transform_tags.inverse_transform(predictor_dt.predict(transform_text_matrix)))
        
      

    if st.button('Recommend'):
         # when no file and no input text are given then print error message
        if uploaded_file is None and len(input_text) <= 0:
            st.error('Upload a file or enter text')
        # when there is a file uploaded then run file_features
        elif uploaded_file is not None:
            file_features()
        else:
            text_features()
  


def main():
    # front end elements of the web page
    
    html_temp = """
        <div style="background-color:blue;padding:10px">
        <h1 style="color:white;text-align:center;">Stackoverflow App</h1>
        </div>
            """
    st.markdown(html_temp, unsafe_allow_html=True)
    predict()



if __name__ == '__main__':
    main()
