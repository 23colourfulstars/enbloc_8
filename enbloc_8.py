import streamlit as st
import pandas as pd
import joblib
import altair as alt

# Load the model
model = joblib.load('enbloc_model_2.pkl')

@st.cache(allow_output_mutation=True)
def load_data():
    # Load the dataset
    return pd.read_csv('enbloc_likelihood_scores_0304.csv')

st.title('En Bloc Prediction App')

df = load_data()

# Visualizations
st.header('Visualizations')

likelihood_count = df.groupby(['District', 'En Bloc Likelihood']).size().reset_index(name='Count')
chart_likelihood = alt.Chart(likelihood_count).mark_bar().encode(
    x=alt.X('District:N', title='District'),
    y=alt.Y('Count:Q', title='Number of Condos'),
    color='En Bloc Likelihood'
)

st.altair_chart(chart_likelihood, use_container_width=True)

# Top en bloc candidates
st.header('Top En Bloc Candidates')
planning_areas = df['Planning Area'].unique()
selected_area = st.selectbox('Select a planning area:', planning_areas)
top_n = 10
top_candidates = df[df['Planning Area'] == selected_area].nlargest(top_n, 'Enbloc Probability')[['Project Name', 'Enbloc Probability', 'En Bloc Likelihood']]
top_candidates.reset_index(drop=True, inplace=True)
top_candidates.index = top_candidates.index + 1
st.write(top_candidates)

# Comparison of multiple condos
st.header('Compare Multiple Condos')
project_names = df['Project Name'].unique()
selected_projects = st.multiselect('Select condos to compare:', project_names)
comparison_df = df[df['Project Name'].isin(selected_projects)][['Project Name', 'Tenure', 'District', 'Enbloc Probability', 'En Bloc Likelihood']]

# Convert Tenure to Freehold or Leasehold
comparison_df['Tenure'] = comparison_df['Tenure'].apply(lambda x: 'Freehold' if x == 1 else 'Leasehold')

comparison_df.set_index('Project Name', inplace=True)
st.write(comparison_df)

# Custom input (for use with data from https://www.edgeprop.sg/)
st.header('Custom Input')
st.markdown("(Based on URA sales data in the last 12 months. Otherwise, based on latest transaction.")

custom_input = {
    'Land Size': st.number_input('Land Size (sqm):', value=1000),
    'Master Plan GFA': st.number_input('Master Plan GFA (sqm):', value=5000),
    'Plot Ratio': st.number_input('Plot Ratio:', value=1.5),
    'Property Type': st.selectbox('Property Type:', options=[(0, 'Apartment'), (1, 'Condominium')], format_func=lambda x: x[1])[0],
    'Tenure': st.selectbox('Tenure:', options=[(0, 'Leasehold'), (1, 'Freehold')], format_func=lambda x: x[1])[0],
    'District': st.number_input('District:', min_value=1, max_value=28, value=1),
    'Age': st.number_input('Age (estimated, in 2043):', value=10),
    'Number of Units': st.number_input('Number of Units:', value=100),
    'Average Price': st.number_input('Average Price* (psf):', value=1000),
    'Historical High': st.number_input('Historical High (psf):', value=1500),
    'Historical Low': st.number_input('Historical Low (psf):', value=650),
    'Distance to MRT': st.number_input('Distance to MRT (KM):', value=0.01)
}


custom_df = pd.DataFrame(custom_input, index=[0])

custom_en_bloc_prediction = model.predict_proba(custom_df)[0][1]

if custom_en_bloc_prediction > 0.5:
    likelihood_tag = "Very Likely"
elif custom_en_bloc_prediction > 0.2:
    likelihood_tag = "Likely"
elif custom_en_bloc_prediction > 0.1:
    likelihood_tag = "Possible"
else:
    likelihood_tag = "Unlikely"

st.subheader("En Bloc Likelihood for Custom Input")
st.write(f"{likelihood_tag} ({custom_en_bloc_prediction*100:.2f}%)")

