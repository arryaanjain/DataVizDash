"""
Statistical modeling components for the application.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def show_statistical_models(df, numeric_cols):
    """Show statistical models and predictions."""
    st.write("### Statistical Models")

    if len(numeric_cols) >= 2:
        # Select target variable
        target_col = st.selectbox("Select target variable", numeric_cols)

        # Select features
        feature_cols = st.multiselect("Select feature variables",
                                     [col for col in numeric_cols if col != target_col],
                                     default=[col for col in numeric_cols[:min(3, len(numeric_cols))]
                                             if col != target_col])

        if feature_cols:
            # Train a simple linear regression model
            st.write("#### Linear Regression Model")

            try:
                # Check for NaN values in target and features
                if df[target_col].isna().any():
                    st.warning(f"⚠️ The selected target variable '{target_col}' contains missing values. Please select a different variable or clean your data.")
                    return

                if df[feature_cols].isna().any().any():
                    # Identify columns with NaN values
                    nan_cols = [col for col in feature_cols if df[col].isna().any()]
                    nan_cols_str = ", ".join([f"'{col}'" for col in nan_cols])
                    st.warning(f"⚠️ The following feature variables contain missing values: {nan_cols_str}. Please select different variables or clean your data.")
                    return

                # Prepare data
                X = df[feature_cols]
                y = df[target_col]

                # Check if we have enough data
                if len(X) < 10:
                    st.warning("⚠️ Not enough data points for reliable modeling. The model requires at least 10 data points.")
                    return

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train model
                model = LinearRegression()
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_test)

                # Evaluate model
                metrics = {
                    'R² Score': r2_score(y_test, y_pred),
                    'Mean Squared Error': mean_squared_error(y_test, y_pred),
                    'Root Mean Squared Error': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'Mean Absolute Error': mean_absolute_error(y_test, y_pred)
                }

                # Display model coefficients
                coef_df = pd.DataFrame({
                    'Feature': feature_cols,
                    'Coefficient': model.coef_
                })

                col1, col2 = st.columns(2)

                with col1:
                    st.write("Model Coefficients")
                    st.dataframe(coef_df)

                with col2:
                    st.write("Model Performance Metrics")
                    metrics_df = pd.DataFrame({
                        'Metric': list(metrics.keys()),
                        'Value': list(metrics.values())
                    })
                    st.dataframe(metrics_df)

                # Plot actual vs predicted values
                fig = px.scatter(
                    x=y_test,
                    y=y_pred,
                    labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                    title='Actual vs Predicted Values'
                )

                # Add diagonal line (perfect predictions)
                min_val = min(y_test.min(), y_pred.min())
                max_val = max(y_test.max(), y_pred.max())
                fig.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        line=dict(color='red', dash='dash'),
                        name='Perfect Prediction'
                    )
                )

                st.plotly_chart(fig, use_container_width=True)

                # Interactive prediction
                st.write("#### Interactive Prediction")
                st.write("Adjust the values below to get a prediction:")

                # Create sliders for each feature
                input_values = {}
                for feature in feature_cols:
                    try:
                        min_val = float(df[feature].min())
                        max_val = float(df[feature].max())
                        mean_val = float(df[feature].mean())

                        step = (max_val - min_val) / 100
                        input_values[feature] = st.slider(
                            f"{feature}",
                            min_value=min_val,
                            max_value=max_val,
                            value=mean_val,
                            step=step
                        )
                    except Exception as e:
                        st.warning(f"⚠️ Could not create slider for feature '{feature}'. Error: {str(e)}")
                        return

                # Make prediction with input values
                input_df = pd.DataFrame([input_values])
                prediction = model.predict(input_df)[0]

                st.write(f"### Predicted {target_col}: **{prediction:.2f}**")

            except ValueError as e:
                error_msg = str(e)
                if "Input contains NaN" in error_msg or "contains NaN" in error_msg:
                    st.warning("⚠️ The selected data contains missing values. Please select different variables or clean your data.")
                elif "Input contains infinity" in error_msg or "infinity" in error_msg:
                    st.warning("⚠️ The selected data contains infinite values. Please select different variables.")
                else:
                    st.warning(f"⚠️ Invalid selection: Please check your data and try again.")
            except Exception as e:
                st.warning("⚠️ An error occurred during model training. Try selecting different variables or check your data for issues.")
        else:
            st.info("Please select at least one feature variable.")
    else:
        st.info("Statistical modeling requires at least two numeric columns.")

def show_advanced_analytics(df, numeric_cols, categorical_cols):
    """Show advanced analytics using tabs."""
    st.subheader("Advanced Analytics")

    # Create tabs for different analytics features
    analytics_tabs = st.tabs(["KPI Dashboard", "Comparative Analysis", "Statistical Models"])

    # KPI Dashboard tab
    with analytics_tabs[0]:
        from analytics.kpi import show_kpi_dashboard
        show_kpi_dashboard(df, numeric_cols)

    # Comparative Analysis tab
    with analytics_tabs[1]:
        from analytics.comparative import show_comparative_analysis
        show_comparative_analysis(df, numeric_cols, categorical_cols)

    # Statistical Models tab
    with analytics_tabs[2]:
        show_statistical_models(df, numeric_cols)
