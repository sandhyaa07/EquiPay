import streamlit as st
import pandas as pd
import numpy as np
import seaboarn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from PyPDF2 import PdfReader
from model import query_gemma2_model  # Import the function from model.py
import os

# Styling the sidebar and the main content
st.markdown(
    """
    <style>

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #FFDB58; /* Light Blue  */
        color: white; /* Text color for sidebar */
    }

    /* Main content area styling */
    [data-testid="stAppViewContainer"] {
        background-color: #FFFFFF; /* White */
    }

    /* Headers and Subheaders in the main content */
    h1, h2, h3, h4, h5, h6 {
        color: #000000; /* Black for headers */
    }

    /* Text elements in the main content */
    p {
        color: #000000; /* Black for paragraph text */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------- STEP 1: Load and Process Legal Documents ---------
def load_documents(pdf_folder):
    documents = []
    for file_name in os.listdir(pdf_folder):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, file_name)
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            documents.append({"content": text, "title": file_name})
    return documents


def split_text_into_chunks(text, chunk_size=512, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks

# Dummy data and models for testing (replace with your real models)
data_appraisal = {
    "Name": ["Alice", "Bob", "Clara", "David", "Ella", "Frank", "Grace", "Henry", "Ivy", "Jack"],
    "Gender": ["Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male"],
    "Department": ["HR", "IT", "IT", "Finance", "Finance", "HR", "IT", "Finance", "HR", "IT"],
    "Role": ["Manager", "Engineer", "Engineer", "Analyst", "Analyst", "Manager", "Engineer", "Analyst", "Manager", "Engineer"],
    "EnvironmentSatisfaction": [3, 2, 4, 1, 3, 2, 3, 4, 3, 2],
    "SalaryHikePercentage": [12, 8, 15, 6, 10, 7, 11, 14, 9, 8],
    "WorkLifeBalance": [3, 2, 4, 1, 3, 2, 3, 4, 3, 2],
    "YearsAtCompany": [5, 3, 6, 2, 4, 3, 5, 7, 6, 3],
    "ExperienceInCurrentRole": [3, 2, 4, 1, 3, 2, 3, 5, 4, 2],
    "YearsSinceLastPromotion": [1, 3, 2, 4, 3, 3, 1, 1, 2, 3],
    "YearsWithCurrentManager": [4, 3, 5, 2, 3, 3, 4, 5, 4, 3],
    "CurrentSalary": [55000, 60000, 58000, 65000, 56000, 62000, 57000, 67000, 54000, 63000],
    "DeservesAppraisal": [1, 0, 1, 0, 1, 0, 1, 1, 1, 0],
    "AppraisalPercentage": [15, 0, 20, 0, 12, 0, 10, 18, 14, 0],
}
df_appraisal = pd.DataFrame(data_appraisal)

features = [
    "EnvironmentSatisfaction",
    "SalaryHikePercentage",
    "WorkLifeBalance",
    "YearsAtCompany",
    "ExperienceInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrentManager",
]

X_appraisal = df_appraisal[features]
y_classification = df_appraisal["DeservesAppraisal"]
y_regression = df_appraisal["AppraisalPercentage"]

classification_model = RandomForestClassifier(random_state=42)
classification_model.fit(X_appraisal, y_classification)

regression_model = RandomForestRegressor(random_state=42)
regression_model.fit(X_appraisal, y_regression)


# Pay Gap Detection Code
def pay_gap_detection():
    st.title("Pay Gap Detection and Salary Prediction")
    
    
    # Create dataset directly within the code
    data = pd.DataFrame({
        "Employee ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Gender": ["Male", "Female", "Male", "Female", "Female", "Male", "Female", "Male", "Female", "Male"],
        "Education Level": ["Bachelor's", "Master's", "Bachelor's", "PhD", "Master's", "Bachelor's", "Bachelor's", "PhD", "Bachelor's", "Master's"],
        "Experience (Years)": [5, 8, 3, 10, 6, 7, 4, 12, 2, 9],
        "Job Role": ["Software Dev", "Data Scientist", "Software Dev", "Data Scientist", "Software Dev",
                     "Software Dev", "Data Scientist", "Data Scientist", "Software Dev", "Software Dev"],
        "Salary": [75000, 85000, 72000, 95000, 77000, 80000, 81000, 105000, 70000, 87000],
        "Department": ["IT", "Data Analytics", "IT", "Data Analytics", "IT", "IT", "Data Analytics",
                       "Data Analytics", "IT", "IT"]
    })
    
    st.header("Dataset")
    st.write(data)

    # Encode categorical variables
    label_encoders = {}
    categorical_features = ['Gender', 'Education Level', 'Job Role', 'Department']
    for feature in categorical_features:
        le = LabelEncoder()
        data[feature] = le.fit_transform(data[feature])
        label_encoders[feature] = le

    # Features and target variable
    X = data.drop(columns=['Employee ID', 'Salary'])
    y = data['Salary']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Model Evaluation")
    st.write(f"Mean Squared Error: {mse}")
    st.write(f"R^2 Score: {r2}")

    # Calculate the average salary by gender
    gender_salary = data.groupby('Gender')['Salary'].mean()

    # Display average salary by gender
    st.subheader("Average Salary by Gender")
    st.write(gender_salary)

    # Calculate the pay gap as the difference between female and male salaries
    if 0 in gender_salary.index and 1 in gender_salary.index:  # 0 might represent Male and 1 Female depending on encoding
        pay_gap = gender_salary[1] - gender_salary[0]
        st.subheader("Pay Gap Detection")
        st.write(f"Pay Gap: ${pay_gap:.2f}")
    else:
        st.write("Gender data not available for both Male and Female.")


    # Prediction Input
    st.subheader("Predict Salary")
    gender = st.selectbox("Gender", label_encoders['Gender'].classes_)
    education = st.selectbox("Education Level", label_encoders['Education Level'].classes_)
    experience = st.slider("Experience (Years)", min_value=0, max_value=20, value=5)
    job_role = st.selectbox("Job Role", label_encoders['Job Role'].classes_)
    department = st.selectbox("Department", label_encoders['Department'].classes_)

    # Prepare input for prediction
    input_data = pd.DataFrame({
        'Gender': [label_encoders['Gender'].transform([gender])[0]],
        'Education Level': [label_encoders['Education Level'].transform([education])[0]],
        'Experience (Years)': [experience],
        'Job Role': [label_encoders['Job Role'].transform([job_role])[0]],
        'Department': [label_encoders['Department'].transform([department])[0]]
    })

    if st.button("Predict"):
        predicted_salary = model.predict(input_data)
        st.write(f"Predicted Salary: ${predicted_salary[0]:.2f}")


# Appraisal Prediction Code
def appraisal_prediction():
    st.title("Employee Appraisal Prediction")

    st.write(
        "Fill in the details below to predict whether the employee deserves an appraisal, "
        "along with the appraisal percentage and amount."
    )

    # Form for input
    with st.form("appraisal_form"):
        environment_satisfaction = st.slider("Environment Satisfaction", 1, 4, 3)
        salary_hike_percentage = st.slider("Salary Hike Percentage", 0, 30, 10)
        work_life_balance = st.slider("Work Life Balance", 1, 4, 3)
        years_at_company = st.slider("Years at Company", 1, 10, 5)
        experience_in_role = st.slider("Experience in Current Role (Years)", 1, 10, 3)
        years_since_promotion = st.slider("Years Since Last Promotion", 0, 5, 2)
        years_with_manager = st.slider("Years with Current Manager", 0, 5, 3)
        performance_rating = st.slider("Performance Rating (1-5)", min_value=1, max_value=5, value=3)
        experience_years = st.slider("Experience (Years)", min_value=0, max_value=30, value=5)
        current_salary = st.number_input("Current Salary (₹)", min_value=10000, value=55000)

        submit_button = st.form_submit_button("Predict")

    # Prediction logic
    if submit_button:
        input_data = np.array([[environment_satisfaction, salary_hike_percentage, work_life_balance,
                                years_at_company, experience_in_role, years_since_promotion, years_with_manager]])

        classification_result = classification_model.predict(input_data)
        regression_result = regression_model.predict(input_data)

        st.subheader("Prediction Results")
        st.write("Employee Deserves Appraisal: ", "Yes" if classification_result[0] == 1 else "No")
        st.write(f"Predicted Appraisal Percentage: {regression_result[0]:.2f}%")

        # Optionally, display predicted salary after appraisal
        predicted_salary = current_salary + (regression_result[0] * current_salary / 100)
        performance_bonus = current_salary * (performance_rating - 1) * 0.05  # 5% per performance rating
        experience_bonus = current_salary * (experience_years / 10) * 0.03 
        total_appraisal = performance_bonus + experience_bonus
        new_salary = current_salary + total_appraisal
        st.write(f"Predicted Salary After Appraisal: ₹{predicted_salary:.2f}")
        st.write(f"Base Salary: ${current_salary:.2f}")
        st.write(f"Performance Bonus: ${performance_bonus:.2f}")
        st.write(f"Experience Bonus: ${experience_bonus:.2f}")
        st.write(f"Total Appraisal: ${total_appraisal:.2f}")
        st.write(f"New Salary after Appraisal: ${new_salary:.2f}")



# Introduction Section
def introduction():
    st.title("Equipay - Pay Gap Detection and Appraisal Management System")
    
    st.write(
        """
        **Equipay** is an AI-powered system designed to detect pay gaps in organizations and predict employee appraisals 
        based on various factors such as experience, performance, and tenure. The system leverages machine learning 
        models to analyze salary trends and appraisal patterns within an organization.
        
        ### Features:
        - **Pay Gap Detection**: Identifies salary disparities based on gender, education, role, and experience.
        - **Salary Prediction**: Provides salary predictions for new employees based on historical data.
        - **Appraisal Prediction**: Predicts whether an employee deserves an appraisal and estimates the percentage of the appraisal.
        - **Legal Chatbot**: Answers questions related to compliance, legal procedures, and employee rights within the organization.
        
        ### Future Scope:
        - **Pay Gap Detection**: Integrate with external databases to analyze pay gaps across various industries and geographies.
        - **Appraisal Management System**: Automate the appraisal process using real-time data and AI models to optimize salary management.

        This system can be used to improve transparency in compensation practices, helping organizations create more equitable work environments.
        """
    )

# --------- STEP 4: Legal Chatbot App ---------
def legal_chatbot():
    st.title("Workplace Legal Issues Chatbot")

    # Upload and process PDFs
    pdf_folder = "legal_documents"
    os.makedirs(pdf_folder, exist_ok=True)
    uploaded_files = st.sidebar.file_uploader("Upload Legal PDFs", accept_multiple_files=True, type=["pdf"])
    for uploaded_file in uploaded_files:
        with open(os.path.join(pdf_folder, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())

    # Load PDFs and split text into chunks
    documents = load_documents(pdf_folder)
    all_chunks = []
    for doc in documents:
        chunks = split_text_into_chunks(doc["content"])
        all_chunks.extend(chunks)

    # User Query
    query = st.text_input("Ask your workplace legal question:")
    if query:
        st.write("Processing your query...")

        # Use top chunks as context (optional)
        context = " ".join(all_chunks[:5])
        prompt = f"Context: {context}\n\nQuery: {query}\n\nResponse:"

        # Query the model
        response = query_gemma2_model(prompt)

        st.subheader("Chatbot Response:")
        st.write(response)

# Streamlit Sidebar for Navigation
def main():
    st.sidebar.title("Menu")
    options = ["Introduction", "Pay Gap Detection & Salary Prediction", "Appraisal Prediction","Legal Chatbot"]
    choice = st.sidebar.radio("Choose Section", options)

    if choice == "Introduction":
        introduction()
    elif choice == "Pay Gap Detection & Salary Prediction":
        pay_gap_detection()
    elif choice == "Appraisal Prediction":
        appraisal_prediction()
    elif choice =="Legal Chatbot":
        legal_chatbot()
        


if __name__ == "__main__":
    main()
