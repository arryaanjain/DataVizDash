"""
Deployment verification module.

This file is used to verify that the latest code is deployed to Streamlit Cloud.
When this file is present and imported, it indicates that the deployment includes
the side-by-side chart feature.
"""

# This timestamp will be updated with each deployment
DEPLOYMENT_TIMESTAMP = "2023-11-15T12:00:00Z"

# This flag indicates that the side-by-side chart feature is included
SIDE_BY_SIDE_CHARTS_ENABLED = True

def verify_deployment():
    """
    Return information about the current deployment.
    
    This function is called by the app to verify that the deployment
    includes the side-by-side chart feature.
    """
    return {
        "timestamp": DEPLOYMENT_TIMESTAMP,
        "side_by_side_charts_enabled": SIDE_BY_SIDE_CHARTS_ENABLED,
        "verification_message": "Side-by-side charts feature is enabled in this deployment."
    }
