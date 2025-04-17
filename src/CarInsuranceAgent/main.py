import sys
package = __import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from tools.see_img import describe_image
import streamlit as st
from car_value_estimation_setup import CarValueEstimatorCrew
from insurance_info_response_setup import InsuranceInfoResponderCrew
from image_car_analysis_setup import ImageCarAnalysisCrew 
from summary_agent import ValuationSummaryCrew


st.title("🚗 Car Insurance Assistant")

task = st.radio("Choose a task", [
    "Estimate Car Value",
    "Ask Insurance Question",
    "Image-Based Car Estimation"
])

if task == "Estimate Car Value":
    car_details = st.text_area("Enter your car details (Make, Model, Year, Condition):")
    if st.button("Estimate Value"):
        if car_details:
            with st.spinner("Estimating..."):
                result = CarValueEstimatorCrew().crew().kickoff(inputs={"car_details": car_details})
                st.success(result)
        else:
            st.warning("Please enter car details.")

elif task == "Ask Insurance Question":
    user_question = st.text_area("Enter your insurance-related question:")
    if st.button("Get Answer"):
        if user_question:
            with st.spinner("Thinking..."):
                result = InsuranceInfoResponderCrew().crew().kickoff(inputs={"user_question": user_question})
                st.success(result)
        else:
            st.warning("Please enter a question.")

elif task == "Image-Based Car Estimation":
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "jfif", "webp"])

    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        st.image(image_bytes, caption="Uploaded Image", use_column_width=True)

        if st.button("Describe Image"):
            with st.spinner("Analyzing image..."):
                try:
                    description = describe_image(image_bytes)
                    st.success("Description:")
                    st.write(description)

                    try:
                        result = ImageCarAnalysisCrew().crew().kickoff(
                            inputs={"image_description": description}
                        )
                        st.success("Agent Result: agent has generated this summary response")
                        with open("report.md", "r") as f:
                            md_content = f.read()

                        # Display it in Streamlit
                        st.markdown(md_content, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Agent error: {e}")
                except Exception as e:
                    st.error(f"Description error: {e}")
