#*********************************************Final Project*****************************************************************************

import pandas as pd
import streamlit as st
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plot
import pytesseract
import plotly.express as px
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist
import spacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk import ne_chunk
from nltk import pos_tag
from nltk.chunk import conlltags2tree, tree2conlltags
from collections import Counter
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import resample

#===========================================================================================================================================
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nlp = spacy.load("en_core_web_sm")
#spacy download en_core_web_sm

def home_page():
    st.subheader(":green[About : This project consists 5 different phases: E- commerce data classification model, E- commerce data visualization, Image Processing, Natural Language Processing, Product Recommendations]")
    st.divider()
    st.subheader(":orange[Objectives of the Project:]")
    st.write("*************************************************")
    st.write(":blue[To create a different plots using given data for Data Visualization]")
    st.write(":blue[To perform a classification model with the given data ]")
    st.write(":blue[To perform a image processing with a uploaded image which are sharpen image, blur image, deblur image, edge detection on image and perform text extraction using OCR]")
    st.write(":blue[To perform tokenization, stemming, key word extraction, sentiment analysis and word cloud for given sentence in input box which is known as Natural Language Processing]")
    st.write(":blue[To create a input box when we enter a product name it must give some products recommendations which are related to that product]")
    st.subheader(":green[-Project done by Dr.BK.Indrani B.Sc., MBA., M.Phil., Ph.D]")
def show_image_page():
    #Image Processing

    st.subheader(":blue[Image Processing]")
    st.write("******************************************************************")

    #sharpening
    def sharpen_image(image):
        kernel = np.array([[-1, -1, -1], [-1, 9, 1], [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened

    #edge detection
    def edge_detection(image):
        edges = cv2.Canny(image, 100, 100)
        return edges

    #blurring
    def blur_image(image):
        blurred=cv2.GaussianBlur(image, (15, 15), 0)
        return blurred

    #deblurred
    def deblur_image(image):
        deblurred = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        deblurred = cv2.cvtColor(deblurred, cv2.COLOR_GRAY2BGR)
        deblurred = cv2.fastNlMeansDenoisingColored(deblurred, None, 10, 10, 7, 21)
        return deblurred

    #text extraction
    def perform_ocr(image):
        text = pytesseract.image_to_string(image)
        return text

    uploaded_file = st.file_uploader(":yellow[Upload an Image]", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        st.image(image, caption='Original image', use_column_width=True)
        option=st.selectbox('select an operator: ', ('original', 'Sharpen', 'Edge Detection', 'Blur', 'Deblur', 'OCR'))

        if option == 'Sharpen':
            Sharpened_image = sharpen_image(image)
            st.image(Sharpened_image, caption='Sharpened Image', use_column_width=True)

        elif option == 'Edge Detection':
            edges = edge_detection(image)
            st.image(edges, caption='Edge Detected Image', use_column_width=True)

        elif option == 'Blur':
            blurred_image = blur_image(image)
            st.image(blurred_image, caption='Blurred Image', use_column_width=True)

        elif option == 'Deblur':
            deblurred_image = deblur_image(image)
            st.image(deblurred_image, caption='Deblurred Image', use_column_width=True)

        elif option == 'OCR':
            ocr_text = perform_ocr(image)
            st.write("Text Extraction from Image")
            st.write(ocr_text)
    st.write(":blue[you are in Image Processing Page]")

#===========================================================================================================================================
def nlp_page():
    st.subheader(":blue[Natural Language Processing]")
    st.write("******************************************************************")

    #Preprocessing Text
    def preprocess_text(text):
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
        return filtered_tokens, stemmed_tokens

    #Extract entities
    def extract_entities(text):
        doc = nlp(text)
        entities = [ent.text for ent in doc.ents]
        return entities

    #Keywords Extraction
    def find_keywords(tokens):
        text = ' '.join(tokens)
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        keywords = [feature_names[idx] for idx in tfidf_matrix.sum(axis=0).argsort()[0, -5:]]
        return keywords

    #Sentiment Analysis
    def perform_sentiment_analysis(text):
        blob = TextBlob(text)
        sentiment = blob.sentiment
        return sentiment

    #Word Cloud
    def display_word_cloud(tokens):
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(tokens))
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title("word cloud")
        st.pyplot(fig)

    input_text = st.text_area("Input Text")
    if st. button("Submit"):
        if input_text:
            st.header(":green[NLP Pre Processing]")
            filtered_tokens, stemmed_tokens = preprocess_text(input_text)

            st.subheader(":yellow[Tokenization]")
            st.write(filtered_tokens)

            st.subheader(":brown[Stemming]")
            st.write(stemmed_tokens)

            st.header(":gray[Named Entity Recognition]")
            named_entities = extract_entities(input_text)
            if named_entities:
                st.write(named_entities)
            else:
                st.write(":orange[No Names Entities found]")

            st.header(":green[Keyword Extraction]")
            keywords = find_keywords(filtered_tokens)
            if keywords:
                st.write(keywords)
            else:
                st.write(":orange[No Keywords found]")

            st.header(":pink[Sentiment Analysis]")
            sentiment = perform_sentiment_analysis(input_text)
            st.write(sentiment)

            st.header(":purple[Word cloud]")
            display_word_cloud(filtered_tokens)
    st.write(":blue[You are in NLP page]")

#===========================================================================================================================================
def e_commerce_page():
    st.subheader(":blue[E Commerce Data Visualization]")
    st.write("******************************************************************")

    def load_data():
        #data = pd.read_csv(r"D:\Ind\projects\Final project\classification_data.csv")
        data = pd.read_csv(r"D:\Ind\projects\Final project\new_df_LE.csv")
        return data

    df = load_data()
    ecom = pd.read_csv(r"D:\Ind\projects\Final project\classification_data.csv")
    sample_data = df

    #Descriptive Analysis
    st.subheader(":green[Central Tendency and Variability of data]")
    st.write(sample_data.describe())

    st.subheader(":orange[Exploratory Analysis]")
    st.divider()

    #Correlation Heatmap
    st.subheader(":green[Correlation Heatmap:]")
    cor = sample_data.corr()
    heatmap_fig, ax = plt.subplots(figsize=(10, 10))
    heatmap = sns.heatmap(cor, annot=True, cmap='plasma', ax=ax)
    st.pyplot(heatmap_fig)
    plt.close()
    st.divider()

    #Scatter Plot
    st.subheader(":green[Scatter Plot with hue based on device browser]")
    scat_jue = sns.scatterplot(data=sample_data, x='count_session', y='count_hit', hue='device_browser')
    st.pyplot(scat_jue.figure)
    st.write("GoogleAnalytics, Safari, Chrome, Edge, Firefox, Samsung Internet, Opera, Android Webview, Apache-HttpClient")
    st.divider()

    #Histogram
    st.subheader(":orange[Distribution of Target Variable]")
    fig = px.histogram(sample_data, x='has_converted', title='Distribution of Target Variable')
    fig.update_layout(xaxis_title='has_converted',
                      yaxis_title='Count',
                      bargap=0.1,
                      showlegend=False)
    st.plotly_chart(fig)
    st.divider()

    #Bubble Chart
    st.subheader(":green[Bubble Chart: Transaction Revenue by device category and Region]")
    bubble_chart = px.scatter(sample_data, x='device_deviceCategory', y='geoNetwork_region',
                              size='transactionRevenue', color='device_deviceCategory',
                              labels={'device_deviceCategory': 'Device Category',
                                      'geoNetwork_region': 'Region',
                                      'transactionRevenue': 'Transaction Revenue'},
                              title='Bubble Chart: Transaction Revenue by Device Category and Region')
    bubble_chart.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')),
                               selector=dict(mode='markers'))
    bubble_chart.update_layout(showlegend=False)
    st.plotly_chart(bubble_chart)
    st.divider()

    #Pie Chart
    st.subheader("Pie Chart for Device Category")
    dev = ecom["device_deviceCategory"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(dev, labels=dev.index, autopct="%1.1f%%", startangle=90)
    ax.axis('equal')
    st.pyplot(fig)
    st.write("Mobile, Desktop, tablet")
    st.divider()

    #Violin Plot
    st.subheader("Violin plot for Device browser and category")
    vio_fig, ax = plt.subplots(figsize=(10, 10))
    sns.violinplot(data=sample_data, x="device_browser", y="device_deviceCategory", ax=ax)
    plt.title('Distribution of device browser by device category')
    plt.xlabel("device browser")
    plt.ylabel("device category")
    plt.xticks(rotation=45)
    st.pyplot(vio_fig)
    st.write("Device-Browser: Android Webview, Apache-HttpClient, Chrome, Edge, Firefox, GoogleAnalytics, Safari, Samsung Internet, Opera")
    st.write("device-category: desktop, mobile, tablet")
    st.write(":blue[you are in ecommerce page]")
    st.divider()

#===========================================================================================================================================
def model_ui_page():
    st.subheader(":blue[E Commerce Data Classification Model]")
    st.write("******************************************************************")

    def load_data():
        data = pd.read_csv(r"D:\Ind\projects\Final project\classification_data.csv")
        return data
    df = load_data()

    encoder = LabelEncoder()
    categorical_cols = ['channelGrouping', 'device_browser', 'device_operatingSystem', 'device_deviceCategory']
    non_numeric_cols = df.select_dtypes(exclude=['float', 'int']).columns

    for col in non_numeric_cols:
        df[col] = encoder.fit_transform(df[col])

    df.dropna(inplace=True)
    df=df.astype(float)

    X = df.drop(columns=['has_converted'])
    y = df['has_converted']
    X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2, random_state=42)

    #Classification models
    models = {'Logistic Regression': LogisticRegression(),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier()}

    model_metrics = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        model_metrics[name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-score': f1}
    st.subheader(':orange[Customer Conversion Prediction]')

    count_session = st.number_input('Count_Session: ', value=0)
    count_hit = st.number_input('Count_Hit: ', value=0)
    device = st.selectbox('Select box: ', df['device_deviceCategory'].unique())
    if st.button('Converted / Not Converted'):
        input_data = pd.DataFrame({'count_session': [count_session],
                                   'count_hit': [count_hit],
                                   'device_deviceCategory': [device]}
                                  )
        input_data = input_data.reindex(columns=X.columns, fill_value=0)

        predictions = {}
        for name, model in models.items():
            predictions[name] = model.predict(input_data)[0]
        st.write("###:orange[Prediction Results: ]")

        for name, result in predictions.items():
            st.write(f'{name}: {"Convert" if result == 1 else "Not Convert"}')
        st.write("###:pink[Model Evaluation Matrix: ]")
        metrics_df = pd.DataFrame(model_metrics).T
        st.write(metrics_df)
    st.write(":blue[you are in e commerce page]")

#===========================================================================================================================================
def recommendation_page():
    st.subheader(" :blue[Product Recommendation]")
    st.write("******************************************************************")

    #Product Recommendations
    def recommend_products(product_name):
        recommendation = {"Mobile": ["Mobile Case", "scratch card", "mobile charger"],
                          "table": ["chair", "table cover", "slippers"],
                          "bottle": ["plates", "tumblers", "bags"],
                          "baby_products": ["milk", "food & juices", "cream"],
                          "baby_bath": ["diapers", "wipes", "baby_soap", "swimming_pants"],
                          "laptop": ["laptop bag", "keyboard skin", "hard disk"],
                          "desktop": ["web cam", "desktop table", "ergonomic chair"]
                          }
        return recommendation.get(product_name, [":green[No recommendations found]"])

    def page():
        product_name = st.text_input("Enter a product name: ")
        if st.button("Get Recommendations"):
            if product_name:
                recommendations = recommend_products(product_name)
                st.subheader(":pink[Recomended products]")
                for product in recommendations:
                    st.write(product)
            else:
                st.write(":brown[please enter a product name]")

    if __name__ == "__main__":
        page()

#=============================================================================================================================================

def main():
    st.header(" :red[Integrated E-Commerce Analytics, Image Processing, Natural Language Processing and Recommendation Engine]", divider="rainbow")
    st.sidebar.title(":blue[Project Navigation]")
    st.sidebar.text("This sidebar helps for Navigation")

#Radio Buttons for Page Navigation
    page = st.sidebar.radio(":green[Select Page to go -->]", ("Home", "Image Processing", "Natural Language Processing", "E-Commerce Data Visualisation", "E-Commerce Classification Model", "Product Recommendation"))

    if page == "Home":
        home_page()
    elif page == "Image Processing":
        show_image_page()
    elif page == "Natural Language Processing":
        nlp_page()
    elif page == "E-Commerce Data Visualisation":
        e_commerce_page()
    elif page == "E-Commerce Classification Model":
        model_ui_page()
    elif page == "Product Recommendation":
        recommendation_page()

if __name__=="__main__":
    main()
#========================================================================================================================================================
