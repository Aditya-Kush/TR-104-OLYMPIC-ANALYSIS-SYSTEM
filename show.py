import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Feature list
features = ['Sex', 'Age', 'Height', 'Weight', 'Year', 'Season', 'Sport']

# Title
st.title("🏅 Olympic Medal Prediction App")
st.markdown("Predict Olympic medals and performance insights based on athlete data.")

# Medal mapping
medal_map = {1: 'Gold', 2: 'Silver', 3: 'Bronze'}

# File upload
uploaded_file = st.file_uploader("Upload CSV with Athlete Data", type="csv")

def performance_score(row):
    # Example: arbitrary weighted sum of Age, Height, Weight (scaled between 0-100)
    score = (0.3 * (row['Age'] / 40) + 0.4 * (row['Height'] / 220) + 0.3 * (row['Weight'] / 150)) * 100
    return round(score, 2)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Clean and encode data
        df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})
        df['Season'] = df['Season'].map({'Summer': 1, 'Winter': 2})
        df['Sport'] = df['Sport'].astype('category').cat.codes

        # Check required features
        if all(col in df.columns for col in features):
            X = df[features]
            X = X.fillna(X.mean())
            X_scaled = scaler.transform(X)

            # Predict medal class and probabilities
            preds = model.predict(X_scaled)
            probs = model.predict_proba(X_scaled)

            # Add predictions
            df['Predicted Medal'] = [medal_map[p] for p in preds]
            df['Confidence'] = np.max(probs, axis=1).round(3)

            # Add probability columns for each medal
            for i, medal_name in medal_map.items():
                df[f'Prob_{medal_name}'] = probs[:, i-1].round(3)

            # Add performance score (demo)
            df['Performance Score'] = df.apply(performance_score, axis=1)

            st.success("Prediction complete!")
            st.dataframe(df[features + ['Predicted Medal', 'Confidence', 'Performance Score',
                                       'Prob_Gold', 'Prob_Silver', 'Prob_Bronze']])

            # Medal prediction count barplot
            pred_counts = df['Predicted Medal'].value_counts()
            st.subheader("🏆 Medal Prediction Count (Bar Chart)")
            fig, ax = plt.subplots()
            sns.barplot(x=pred_counts.index, y=pred_counts.values, ax=ax, palette='viridis')
            ax.set_ylabel("Count")
            ax.set_title("Predicted Medal Distribution")
            st.pyplot(fig)

            # Medal prediction distribution pie chart
            st.subheader("🏆 Medal Prediction Distribution (Pie Chart)")
            fig2, ax2 = plt.subplots()
            ax2.pie(pred_counts.values, labels=pred_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('viridis', len(pred_counts)))
            ax2.axis('equal')  # Equal aspect ratio ensures pie chart is circular.
            st.pyplot(fig2)

            # Boxplots of Age, Height, Weight by Predicted Medal
            st.subheader("Distribution of Age, Height, and Weight by Predicted Medal")
            fig3, axes = plt.subplots(1, 3, figsize=(18,5))
            sns.boxplot(x='Predicted Medal', y='Age', data=df, ax=axes[0], palette='Set2')
            axes[0].set_title("Age Distribution")
            sns.boxplot(x='Predicted Medal', y='Height', data=df, ax=axes[1], palette='Set2')
            axes[1].set_title("Height Distribution")
            sns.boxplot(x='Predicted Medal', y='Weight', data=df, ax=axes[2], palette='Set2')
            axes[2].set_title("Weight Distribution")
            st.pyplot(fig3)

            # Confidence score histogram
            st.subheader("Confidence Score Distribution")
            fig4, ax4 = plt.subplots()
            sns.histplot(df['Confidence'], bins=30, kde=True, color='skyblue', ax=ax4)
            ax4.set_xlabel('Confidence Score')
            ax4.set_ylabel('Frequency')
            st.pyplot(fig4)

            # Scatter plot of Performance Score vs Confidence colored by Predicted Medal
            st.subheader("Performance Score vs Confidence by Predicted Medal")
            fig5, ax5 = plt.subplots()
            sns.scatterplot(data=df, x='Performance Score', y='Confidence', hue='Predicted Medal', palette='dark', s=80, ax=ax5)
            ax5.set_xlabel('Performance Score')
            ax5.set_ylabel('Confidence')
            st.pyplot(fig5)

        else:
            st.error(f"Input CSV must contain columns: {features}")

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("Please upload a CSV file to begin.")
