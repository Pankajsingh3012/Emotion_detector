import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_select import image_select
import base64
import cv2
import pandas as pd
import pywhatkit

# set title of page
# st.markdown("<h1 '><font color='grey' ><center>Emotion Detector App</center></h1>",unsafe_allow_html=True)


# # ctreating gride of image to select
# img = image_select(
#     label="_Select a image to predict_",
#     images=[
#         "images/1.jpg",
#         "images/2.jpeg",
#         "images/3.jpeg",
#         "images/4.jpg"
#
#     ],
#
# )
#
# # image select for background
# titleimg = "emoji-emotions.gif"
#
# #impliment background formating
# def set_bg_hack(main_bg):
#     # set bg name
#     main_bg_ext = "gif"
#     st.markdown(
#         f"""
#          <style>
#          .stApp {{
#              background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
#              background-repeat: no-repeat;
#              background-position: right 50% bottom 95% ;
#              background-size: contain;
#              background-attachment: scroll;
#          }}
#          </style>
#          """,
#         unsafe_allow_html=True,
#     )
#
# set_bg_hack(titleimg)
#
# uploaded_img = st.file_uploader("Upload an image file",
#                                 type  = ["png", "jpg", "jpeg"])
#
#
# # load model
# @st.cache_resource
# def cache_model(model_add):
#     model = tf.keras.models.load_model(model_add)
#     return model
#
#
# model = cache_model("emotion_detector")
#
# # creating predict button
# predict = st.button("Predict")
#
# # defining harcascade classifier and class_names
# face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# class_names = ["Angry", "Happy", "Sad"]
#
#
# def model_pred(model, image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     results = face_detector.detectMultiScale(gray, scaleFactor=1.05,
#                                              minNeighbors=10,
#                                              minSize=(100, 100))
#     if len(results) != 0:
#         for x, y, w, h in results:
#             img_crp = image[y:y + h, x:x + w]
#             img_crp = cv2.resize(img_crp, (350, 350))
#             y_pred_prob = model.predict(tf.expand_dims(img_crp,
#                                                        axis=0))
#             y_pred = np.argmax(y_pred_prob, axis=-1)
#             # print(y_pred_prob)
#             label = class_names[int(y_pred)]
#             cv2.rectangle(image, (x, y), (x + w, y + h),
#                           color=(0, 255, 0),
#                           thickness=10)
#             cv2.putText(image, f"{label},{round(round(np.max(y_pred_prob), 2)*100,2)}%",
#                         (x, y + h), cv2.FONT_HERSHEY_COMPLEX, 2,
#                         (0, 255, 255), 2)
#     return image
#
#
# if predict:
#     if uploaded_img:
#         # img_array = np.array(uploaded_img)
#         img_array = np.array(Image.open(uploaded_img))
#         result_img = model_pred(model, img_array)
#         st.image(result_img)
#
#     else:
#         st.write("Please upload a valid image")
#
# else:
#     image_array = np.array(Image.open(img))
#     result_img = model_pred(model, image_array)
#     st.image(result_img)
#
# # st.write("enter phone number.")
#number = st.text_area("Enter your mobile number")

df = pd.read_csv("mobile.csv")
with open("mobile.csv","a+") as f:

    if 9982532962 in df.mobile.values:
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        # set up the SMTP server
        smtp_server = smtplib.SMTP('smtp.gmail.com', 587)
        smtp_server.starttls()

        # Login Credentials for sending the mail
        email_address = "ardhpankaj@gmail.com"
        password = "Pankaj@3012"

        # setup the parameters of the message
        msg = MIMEMultipart()
        msg['From'] = email_address
        msg['To'] = "bhavyachoubisa02@gmail.com"
        msg['Subject'] = "Subject of the Mail"

        # add in the message body
        message = "Hello, this is a test email from Python."
        msg.attach(MIMEText(message, 'plain'))

        # send the message via the server
        smtp_server.login(email_address, password)
        smtp_server.send_message(msg)
        smtp_server.quit()
        smtplib.SMTPAuthenticationError: (535, '5.7.3 Authentication unsuccessful')
        # we import the Twilio client from the dependency we just installed
        # from twilio.rest import Client
    #
    #     # the following line needs your Twilio Account SID and Auth Token
    #     client = Client("AC2e260fdb51eba3bab9216becd62408af", "21d42db487c036f905a24f3db19c4c05")
    #
    #     # change the "from_" number to your Twilio number and the "to" number
    #     # to the phone number you signed up for Twilio with, or upgrade your
    #     # account to send SMS to any phone number
    #     client.messages.create(to="+917357212221",
    #                            from_="+16592742609",
    #                            body="from emotion_detector")
    # else:
    #     f.write("8003206434\n")

# df = pd.read_csv("mobile.csv")
# for x in df.mobile.values:
#     print(x)
#CUKFPZTJJYE8H7PPPDL3VGKZ