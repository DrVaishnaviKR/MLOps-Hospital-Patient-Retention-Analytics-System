import pandas as pd


def engineer_features(df, reference_date=None):
    """
    Engineer features from raw patient data.
    Uses a fixed reference_date so training and inference produce
    consistent Recency_Days values.
    """
    df = df.copy()

    # Drop PatientID (not a feature)
    if "PatientID" in df.columns:
        df.drop(columns=["PatientID"], inplace=True)

    # Convert Last_Interaction_Date to Recency_Days
    if "Last_Interaction_Date" in df.columns:
        df["Last_Interaction_Date"] = pd.to_datetime(
            df["Last_Interaction_Date"],
            dayfirst=True,
            errors="coerce"
        )

        # Use a fixed reference date for consistency between
        # training and inference (latest date in training data + 1 day)
        if reference_date is None:
            reference_date = df["Last_Interaction_Date"].max() + pd.Timedelta(days=1)

        df["Recency_Days"] = (
            reference_date - df["Last_Interaction_Date"]
        ).dt.days

        df.drop(columns=["Last_Interaction_Date"], inplace=True)

    # Engagement Ratio
    if "Visits_Last_Year" in df.columns and "Tenure_Months" in df.columns:
        df["Engagement_Ratio"] = (
            df["Visits_Last_Year"]
            / df["Tenure_Months"].replace(0, 1))
    # Missed appointment ratio
    if "Missed_Appointments" in df.columns and "Visits_Last_Year" in df.columns:
        df["Missed_Ratio"] = (
            df["Missed_Appointments"]
            / df["Visits_Last_Year"].replace(0, 1))
    # Average satisfaction score
    sat_cols = [
        "Overall_Satisfaction",
        "Wait_Time_Satisfaction",
        "Staff_Satisfaction"
    ]
    existing_sat = [c for c in sat_cols if c in df.columns]
    if existing_sat:
        df["Avg_Satisfaction"] = df[existing_sat].mean(axis=1)

    return df
