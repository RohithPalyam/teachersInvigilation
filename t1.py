import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.metric_cards import *
import pymongo
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
from deepface import DeepFace
import pyzbar.pyzbar as pyzbar
import cv2
from datetime import datetime

# Initialize session state
session_keys = ['Id', 'Login Status']
for key in session_keys:
    if key not in st.session_state:
        st.session_state[key] = None

class Teachers:
    def __init__(self):
        """Initialize database connections."""
        self.client = pymongo.MongoClient(st.secrets["database"]["clientlink"])
        self.scheduledDB = self.client["ScheduledExams"]
        self.studentsDB = self.client["StudentsDB"]
        self.studentsCollection = self.studentsDB['StudentsCollection']
        self.teachersDB = self.client["TeachersDB"]
        self.departmentsDB = self.client["DepartmentsDB"]
        self.roomsDB = self.client["RoomsDB"]
        self.validationDB = self.client["validationDB"]
        self.invigilatorDB = self.client["InvigilatorDB"]
        self.teachersCollection = self.teachersDB["TeachersCollection"]
        self.complaintsDB=self.client["ComplaintsDB"]

    def login(self):
        """Handles teacher login."""
        st.header("🔑 Login")
        col1, col2 = st.columns([1, 2],border=True)

        with col1:
            invigilation_id = st.text_input("Enter Invigilation ID")
            if st.button("Login", use_container_width=True, type='primary'):
                teacher_details = self.teachersCollection.find_one({"empID": invigilation_id}, {"_id": 0})
                if teacher_details:
                    st.session_state['Login Status'] = True
                    st.session_state['Id'] = invigilation_id
                    with col2:
                        st.success("✅ Login Successful!")
                        for key, value in teacher_details.items():
                            st.write(f"**{key}:** {value}")
                else:
                    st.error("❌ Invalid Invigilation ID. Please try again.")

    def invigilation_duties(self):
        """Fetches and displays invigilation duties."""
        st.header("📋 Invigilation Duties")
        col1, col2 = st.columns([1, 2],border=True)

        with col1:
            collections = self.invigilatorDB.list_collection_names()
            if collections:
                selected_collection = st.selectbox("Select Invigilation Type", collections)
                if st.button("Fetch Duties", use_container_width=True, type='primary'):
                    if st.session_state['Id']:
                        invigilations = list(self.invigilatorDB[selected_collection].find(
                            {"invigilatorID": st.session_state['Id']}, {"_id": 0}
                        ))
                        num_invigilations = len(invigilations)
                        with col2:
                            if num_invigilations == 0:
                                st.info("🚀 No invigilation assigned, enjoy your day!")
                            else:
                                st.metric("Total Invigilations", value=str(num_invigilations))
                                st.dataframe(pd.DataFrame(invigilations))
                    else:
                        st.warning("⚠️ Please login first.")
            else:
                st.warning("⚠️ No invigilation data found.")

    def validate_face(self, col1, col2, select_hall_ticket_number,collection,subject):
        col2.subheader("You Are Performing Face Validation", divider='blue')
        captured_image = col2.camera_input("Take a photo for validation")
        
        if captured_image:
            col2.image(captured_image, caption="Captured Image", use_container_width=True)
            if col2.checkbox("Verify"):
                student_data = self.studentsCollection.find_one({"roll_number": select_hall_ticket_number}, {"front_photo": 1, "_id": 0})
                
                if student_data and "front_photo" in student_data:
                    stored_image_data = student_data['front_photo']
                    stored_image = Image.open(BytesIO(stored_image_data)).convert('RGB')
                    col2.image(stored_image, caption="Student's Stored Image", use_container_width=True)
                    
                    taken_image = Image.open(captured_image).convert('RGB')
                    taken_image = np.array(taken_image)
                    stored_image = np.array(stored_image)
                    
                    models = ["VGG-Face", "Facenet", "OpenFace", "DeepID", "SFace"]
                    results = []
                    
                    for model in models:
                        try:
                            result = DeepFace.verify(taken_image, stored_image, model_name=model)
                            results.append((model, result['verified']))
                        except Exception as e:
                            results.append((model, str(e)))
                    
                    df_results = pd.DataFrame(results, columns=["Model", "Result"])
                    col2.dataframe(df_results)
                    
                    true_count = df_results['Result'].sum()
                    false_count = len(df_results) - true_count
                    validation_collection = collection.replace("-Schedule", "-Validations")
                    validation_collection=self.validationDB[validation_collection]
                    if true_count > false_count:
                        validation_collection.update_one({"hall_ticket_number": select_hall_ticket_number, "subject": subject},
                        {"$set":{"studentFaceRecognitionStatus":True}})
                        col2.success("✅ Student is Present")
                    else:
                        validation_collection.update_one({"hall_ticket_number": select_hall_ticket_number, "subject": subject},
                        {"$set":{"studentFaceRecognitionStatus":False}})
                        col2.error("❌ Student Verification Failed")
                else:
                    col2.error("No stored photo found for this student.")

    def validation(self):
        col1, col2 = st.columns([1, 2],border=True)
        col1.header("Please Enter Details For", divider='blue')
        
        collections = self.scheduledDB.list_collection_names()
        if collections:
            collection = col1.selectbox("Select the invigilation", collections)
            schedule_collection = self.scheduledDB[collection]

            subject = col1.selectbox("Select the subject", schedule_collection.distinct("subject_name"))

            documents = self.scheduledDB[collection].find_one(
                {"subject_name": subject, "invigilator_ids": {"$in": [st.session_state.get('Id', '')]}}
            )

            if documents and 'room_details' in documents:
                index=documents['invigilator_ids'].index(st.session_state['Id'])
                final_document = documents['room_details'][index]
                
                # Debugging: Check if correct roll numbers are fetched
                hall_ticket_numbers = final_document.get('hallTicketNumbers', [])

                if hall_ticket_numbers:
                    select_hall_ticket_number = col1.selectbox(
                        "Select the hall ticket number", 
                        hall_ticket_numbers, 
                        key=f"hall_ticket_{subject}"  # Unique key for dynamic update
                    )
                    
                    if col1.checkbox("Validate"):
                        col2.header("Validations", divider='blue')
                        options = col2.selectbox("Select Validation Option", ["Validate Face", "Validate Thumb", "Validate QR"])
                        if options == "Validate Face":
                            self.validate_face(col1, col2, select_hall_ticket_number,collection,subject)
                        if options=="Validate QR":
                            self.validateQR(col1, col2, select_hall_ticket_number, collection, subject)
                        if options=="Validate Thumb":
                            self.validateThumb(col1, col2, select_hall_ticket_number, collection, subject)
                else:
                    col1.error("No hall ticket numbers available.")
            else:
                col1.error("No records found.")
        else:
            col1.warning("No scheduled exams found.")
    def validateQR(self, col1, col2, select_hall_ticket_number, collection, subject):
        """Validates student using QR code."""
        col2.subheader("You Are Performing QR Validation", divider='blue')
        uploaded_image = col2.file_uploader("Upload QR Code Image", type=["png", "jpg", "jpeg"])
        
        if uploaded_image:
            image = Image.open(uploaded_image).convert('RGB')
            image_np = np.array(image)
            decoded_objects = pyzbar.decode(image_np)
            
            if decoded_objects:
                decoded_data = decoded_objects[0].data.decode('utf-8')
                col2.success(f"✅ QR Code Data: {decoded_data}")
                if decoded_data.split("-")[-1] == select_hall_ticket_number:
                    validation_collection = collection.replace("-Schedule", "-Validations")
                    validation_collection = self.validationDB[validation_collection]
                    validation_collection.update_one({"hall_ticket_number": select_hall_ticket_number, "subject": subject},
                    {"$set": {"studentQRCodeStatus": True}})
                    col2.success("✅ Student QR Validation Successful")
                else:
                    validation_collection = collection.replace("-Schedule", "-Validations")
                    validation_collection = self.validationDB[validation_collection]
                    validation_collection.update_one({"hall_ticket_number": select_hall_ticket_number, "subject": subject},
                    {"$set": {"studentQRCodeStatus": False}})
                    col2.error("❌ QR Code does not match Hall Ticket Number")
            else:
                col2.error("❌ No QR Code detected in the uploaded image.")

    def check_validations(self):
        """Checks for validations and displays the results."""
        col1, col2 = st.columns([1, 2], border=True)
        col1.subheader("Check Your Validations")

        validation_collections = self.validationDB.list_collection_names()
        
        if not validation_collections:
            col1.warning("No validation collections found.")
            return

        selectedCollection = col1.selectbox("Select validation", validation_collections)

        if selectedCollection:
            col1.write(f'You selected: {selectedCollection}')
            collection = self.validationDB[selectedCollection]

            # Extract the corresponding scheduled exam collection name
            schedule_collection_name = selectedCollection.replace("-Validations", "-Schedule")
            
            if schedule_collection_name not in self.scheduledDB.list_collection_names():
                col1.error("Corresponding schedule collection not found.")
                return

            schedule_collection = self.scheduledDB[schedule_collection_name]

            # Fetch subjects
            subjects = schedule_collection.distinct("subject_name")
            if not subjects:
                col1.error("No subjects found in the schedule.")
                return

            subject = col1.selectbox("Select the subject", subjects)

            # Fetch documents where the logged-in teacher is an invigilator
            teacher_id = st.session_state.get('Id')
            if not teacher_id:
                col1.warning("Please login first.")
                return

            document = schedule_collection.find_one(
                {"subject_name": subject, "invigilator_ids": teacher_id}
            )

            if not document or 'room_details' not in document:
                col1.warning("No room details found for this subject.")
                return

            # Fetch the invigilator's assigned room index
            try:
                index = document['invigilator_ids'].index(teacher_id)
                final_document = document['room_details'][index]
            except ValueError:
                col1.error("You are not assigned to this exam.")
                return

            # Fetch hall ticket numbers
            hall_ticket_numbers = final_document.get('hallTicketNumbers', [])
            if not hall_ticket_numbers:
                col1.warning("No hall ticket numbers available.")
                return
            collection = self.validationDB[selectedCollection]
            parentList=[]
            for i in hall_ticket_numbers:
                document = collection.find_one(
                    {"hall_ticket_number": i, "subject": subject},
                {
                    "_id": 0,
                    "studentBooketNumber": 1,
                    "hall_ticket_number": 1,
                    "studentName": 1,
                    "studentQRCodeStatus": 1,
                    "studentFaceRecognitionStatus": 1,
                    "studentThumbStatus": 1,
                    "StudentsFinalStatus": 1})
                parentList.append(document)
            col2.info("Here is the validation details")
            col2.dataframe(parentList)

    def validateThumb(self, col1, col2, select_hall_ticket_number, collection, subject):
        """Validates student using fingerprint recognition with OpenCV."""
        col2.subheader("You Are Performing Thumb Validation", divider='blue')

        uploaded_reference = self.studentsCollection.find_one({'roll_number': select_hall_ticket_number})
        uploaded_scan = col2.file_uploader("Upload Scanned Fingerprint (BMP)", type=["bmp"], key="scan")

        if uploaded_reference and uploaded_scan:
            if "left_thumb" in uploaded_reference:
                try:
                    # Convert binary data to file-like object
                    binary_data = uploaded_reference["left_thumb"]
                    ref_image = Image.open(BytesIO(binary_data)).convert('L')
                    scan_image = Image.open(uploaded_scan).convert('L')

                    ref_array = np.array(ref_image)
                    scan_array = np.array(scan_image)

                    # Debugging prints
                    print(f"Reference Image Shape: {ref_array.shape}")
                    print(f"Scan Image Shape: {scan_array.shape}")

                    # Perform matching
                    score = self.match_fingerprints(ref_array, scan_array)
                    col2.metric("Matching Score", value=round(score, 2))

                    validation_collection = collection.replace("-Schedule", "-Validations")
                    validation_collection = self.validationDB[validation_collection]

                    if score >= 2:  # Match threshold
                        validation_collection.update_one(
                            {"hall_ticket_number": select_hall_ticket_number, "subject": subject},
                            {"$set": {"studentThumbStatus": True}}
                        )
                        col2.success("✅ Thumbprint Matched. Student Verified.")
                    else:
                        validation_collection.update_one(
                            {"hall_ticket_number": select_hall_ticket_number, "subject": subject},
                            {"$set": {"studentThumbStatus": False}}
                        )
                        col2.error("❌ Thumbprint Mismatch. Verification Failed.")

                except Exception as e:
                    col2.error(f"Error in processing fingerprints: {str(e)}")
                    print(f"Exception: {e}")  # Debugging
            else:
                col2.error("No left thumb image found in the reference record.")
        else:
            col2.error("Student record or uploaded scan not found.")

    def match_fingerprints(self, img1_array, img2_array):
        """Match fingerprints using SIFT (preferred) or ORB and FLANN-based matcher."""
        
        try:
            sift = cv2.SIFT_create()
            use_sift = True
        except AttributeError:
            sift = None
            use_sift = False

        if use_sift:
            kp1, des1 = sift.detectAndCompute(img1_array, None)
            kp2, des2 = sift.detectAndCompute(img2_array, None)
            index_params = dict(algorithm=1, trees=5)  # KD-tree
        else:
            orb = cv2.ORB_create()
            kp1, des1 = orb.detectAndCompute(img1_array, None)
            kp2, des2 = orb.detectAndCompute(img2_array, None)
            index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)  # LSH-based

        search_params = dict(checks=50)
        
        if des1 is None or des2 is None:
            return 0  # Return score 0 if no features are detected

        if not use_sift:
            des1 = np.asarray(des1, dtype=np.float32)
            des2 = np.asarray(des2, dtype=np.float32)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
        match_score = (len(good_matches) / max(len(kp1), len(kp2))) * 100

        return match_score

    def makeComplaints(self):
        col1, col2 = st.columns([1, 2],gap="medium",border=True)
        col1.subheader("Check Your Validations")

        # List validation collections
        validation_collections = self.validationDB.list_collection_names()
        if not validation_collections:
            col1.warning("No validation collections found.")
            return

        selectedCollection = col1.selectbox("Select validation", validation_collections)
        if not selectedCollection:
            return

        collection = self.validationDB[selectedCollection]
        scheduled_collection_name = selectedCollection.replace("-Validations", "-Schedule")

        # Check if schedule collection exists
        if scheduled_collection_name not in self.scheduledDB.list_collection_names():
            col1.error("Corresponding schedule collection not found.")
            return

        scheduled_collection = self.scheduledDB[scheduled_collection_name]
        subjects = scheduled_collection.distinct("subject_name")
        
        selected_subject = col1.selectbox("Select Subject", subjects)
        if not selected_subject:
            return

        hall_ticket_numbers = scheduled_collection.distinct("hall_ticket_numbers", {"subject_name": selected_subject})
        selected_hall_ticket = col1.selectbox("Select Hall Ticket Number", hall_ticket_numbers)
        if not selected_hall_ticket:
            return

        complaints_collection_name = selectedCollection.replace("-Validations", "-Complaints")
        complaints_collection = self.complaintsDB[complaints_collection_name]

        # Check if complaint exists for the selected hall ticket number and subject
        existing_complaint = complaints_collection.find_one({
            "hall_ticket_number": selected_hall_ticket,
            "subject": selected_subject
        })

        if existing_complaint:
            col2.success("Complaint already exists. You can update or delete it.")
            complaint_text = col2.text_area("Update Complaint", existing_complaint.get("complaints", ""))
            update_button, delete_button = col2.columns(2)
            
            if update_button.button("Update Complaint",use_container_width=True):
                complaints_collection.update_one(
                    {"hall_ticket_number": selected_hall_ticket, "subject": selected_subject},
                    {"$set": {
                        "complaints": complaint_text,
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "time": datetime.now().strftime("%H:%M:%S")
                    }}
                )
                col2.success("Complaint updated successfully!")
            
            if delete_button.button("Delete Complaint",use_container_width=True):
                complaints_collection.delete_one({
                    "hall_ticket_number": selected_hall_ticket,
                    "subject": selected_subject
                })
                col2.warning("Complaint deleted successfully!")
        else:
            col2.info("No complaint found. Please enter your complaint.")
            complaint_text = col2.text_area("Write your complaint")
            if col2.button("Submit Complaint"):
                complaints_collection.insert_one({
                    "hall_ticket_number": selected_hall_ticket,
                    "subject": selected_subject,
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "complaints": complaint_text
                })
                col2.success("Complaint submitted successfully!")


        
    def display(self):
        st.set_page_config(page_title="Admin Panel", layout="wide")
        with st.sidebar:
            selected = option_menu("Navigation", ["Login", "Invigilation Duties", "Validations","check validations","make complaints"], default_index=0)
        if selected == "Login":
            self.login()
        elif selected == "Invigilation Duties":
            self.invigilation_duties()
        elif selected == "Validations":
            self.validation()
        elif selected=="check validations":
            self.check_validations()
        elif selected == "make complaints":
            self.makeComplaints()

teacher_app = Teachers()
teacher_app.display()
