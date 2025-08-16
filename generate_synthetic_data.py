import numpy as np
import pandas as pd

def generate_synthetic_data(n, force_medical_zero=False):
    # Generate synthetic data for n individuals.
    ages = np.random.randint(20, 81, size=n)              # Age: 20-80 years
    heights = np.random.randint(150, 191, size=n)           # Height: 150-190 cm
    weights = np.random.randint(45, 101, size=n)            # Weight: 45-100 kg

    # Lifestyle factors
    smoking = np.random.choice([0, 1, 2, 3], size=n)        # 0: None, 1: Light, 2: Moderate, 3: Heavy
    alcohol = np.random.choice([0, 1, 2, 3], size=n)        # 0: None, 1: Light, 2: Moderate, 3: Heavy
    bp_sys = np.random.randint(100, 161, size=n)            # Systolic BP: 100-160 mmHg
    bp_dia = np.random.randint(60, 101, size=n)             # Diastolic BP: 60-100 mmHg
    pulse = np.random.randint(60, 101, size=n)              # Pulse: 60-100 bpm
    exercise = np.random.choice([0, 1], size=n)             # 1: Yes, 0: No
    water_intake = np.round(np.random.uniform(1.0, 4.0, size=n), 1)  # Water intake in liters
    sleep = np.random.randint(4, 13, size=n)                # Sleep hours: 4-12 hours
    education = np.random.choice([0, 1], size=n, p=[0.3, 0.7])  # 1: Educated, 0: Not Educated
    income = np.random.randint(5000, 100001, size=n)        # Monthly income in INR

    # Medical History (binary indicators)
    if force_medical_zero:
        hypertension = np.zeros(n, dtype=int)
        diabetes = np.zeros(n, dtype=int)
        thyroid = np.zeros(n, dtype=int)
        mental_health = np.zeros(n, dtype=int)
        fatty_liver = np.zeros(n, dtype=int)
    else:
        hypertension = np.random.choice([0, 1], size=n, p=[0.7, 0.3])
        diabetes = np.random.choice([0, 1], size=n, p=[0.85, 0.15])
        thyroid = np.random.choice([0, 1], size=n, p=[0.9, 0.1])
        mental_health = np.random.choice([0, 1], size=n, p=[0.9, 0.1])
        fatty_liver = np.random.choice([0, 1], size=n, p=[0.9, 0.1])

    # Baseline life expectancy
    baseline = 70

    # Derived Measurements
    height_m = heights / 100.0
    bmi = weights / (height_m ** 2)

    # Age-dependent BMI Adjustment
    def compute_bmi_adjustment(age, bmi_val):
        if age >= 65:
            if bmi_val < 23:
                return -2
            elif bmi_val < 30:
                return 0
            else:
                return -10
        else:
            if bmi_val < 18.5:
                return -3
            elif bmi_val < 25:
                return 0
            elif bmi_val < 30:
                return -5
            else:
                return -10
    bmi_adj = np.vectorize(compute_bmi_adjustment)(ages, bmi)

    # Smoking Adjustment (penalty)
    smoking_adj = np.where(smoking == 1, 2, 0) \
                  + np.where(smoking == 2, 4, 0) \
                  + np.where(smoking == 3, 8, 0)
    smoking_adj = -smoking_adj

    # Alcohol Adjustment (penalty)
    alcohol_adj = np.where(alcohol == 1, 2, 0) \
                  + np.where(alcohol == 2, 4, 0) \
                  + np.where(alcohol == 3, 6, 0)
    alcohol_adj = -alcohol_adj

    # Exercise Adjustment (bonus)
    exercise_adj = np.where(exercise == 1, 5, 0)

    # Water Intake Adjustment (penalty if below optimal 2 liters)
    water_adj = np.where(water_intake < 2, 2, 0)
    water_adj = -water_adj

    # Sleep Adjustment using U-shaped effect (optimal ~7.5 hours)
    sleep_adj = -np.abs(sleep - 7.5)

    # Education Adjustment
    education_adj = np.where(education == 1, 3, -2)

    # Income Adjustment (for individuals aged 23 and above)
    income_adj = np.where(ages < 23, 0, np.where(income < 20000, -5, np.where(income > 50000, 3, 0)))

    # Blood Pressure Adjustment (non-linear piecewise for systolic BP)
    def bp_adjust(bp):
        if bp <= 120:
            return 0
        elif bp <= 140:
            return (bp - 120) / 10.0
        else:
            return 2 + (bp - 140) / 5.0
    bp_adjust_vec = np.vectorize(bp_adjust)
    bp_adj = -bp_adjust_vec(bp_sys)

    # Pulse Rate Adjustment (penalty: for every 5 bpm above 70, subtract 1 year)
    pulse_adj = -np.maximum(0, (pulse - 70) // 5)

    # Medical History Adjustments (penalties)
    hypertension_adj = np.where(hypertension == 1, 4, 0)
    diabetes_adj = np.where(diabetes == 1, 5, 0)
    thyroid_adj = np.where(thyroid == 1, 2, 0)
    mental_health_adj = np.where(mental_health == 1, 3, 0)
    fatty_liver_adj = np.where(fatty_liver == 1, 3, 0)
    med_adj = -(hypertension_adj + diabetes_adj + thyroid_adj + mental_health_adj + fatty_liver_adj)

    # Interaction Effect: If heavy smoker (3) and has diabetes, subtract extra 3 years.
    interaction_adj = -np.where((smoking == 3) & (diabetes == 1), 3, 0)

    # New Bonus: If a person exercises, does not smoke or drink, and has no diseases, award +10 years.
    bonus = np.where(
        (exercise == 1) & (smoking == 0) & (alcohol == 0) &
        (hypertension == 0) & (diabetes == 0) & (thyroid == 0) &
        (mental_health == 0) & (fatty_liver == 0),
        10, 0
    )

    # Total Adjustments and Life Expectancy Calculation
    total_adj = (bmi_adj + smoking_adj + alcohol_adj + exercise_adj + water_adj +
                 sleep_adj + education_adj + income_adj + bp_adj + pulse_adj +
                 med_adj + interaction_adj + bonus)
    predicted_life_expectancy = baseline + total_adj

    # Ensure predicted life expectancy is at least current age + 5 years
    lower_bound = ages + 5
    final_life_expectancy = np.maximum(predicted_life_expectancy, lower_bound)
    final_life_expectancy = np.clip(final_life_expectancy, None, 100)

    df = pd.DataFrame({
        'Age': ages,
        'Height_cm': heights,
        'Weight_kg': weights,
        'BMI': np.round(bmi, 2),
        'Smoking': smoking,
        'Alcohol': alcohol,
        'BP_Systolic': bp_sys,
        'BP_Diastolic': bp_dia,
        'Pulse': pulse,
        'Exercise': exercise,
        'Water_Intake_L': water_intake,
        'Sleep_Hours': sleep,
        'Education': education,
        'Income_INR': income,
        'Hypertension': hypertension,
        'Diabetes': diabetes,
        'Thyroid': thyroid,
        'Mental_Health': mental_health,
        'Fatty_Liver': fatty_liver,
        'Predicted_Life_Expectancy': np.round(predicted_life_expectancy, 1),
        'Final_Life_Expectancy': np.round(final_life_expectancy, 1)
    })
    return df

np.random.seed(42)

# Generate Group 1: 150 records with medical history set to zero
group1 = generate_synthetic_data(150, force_medical_zero=True)
group1.to_csv("synthetic_group1.csv", index=False)

# Generate Group 2: 250 records with all parameters active
group2 = generate_synthetic_data(250, force_medical_zero=False)
group2.to_csv("synthetic_group2.csv", index=False)

# Generate Group 3: 100 records that satisfy the bonus condition
# Start by forcing medical history to zero, then override lifestyle to force bonus conditions:
group3 = generate_synthetic_data(100, force_medical_zero=True)
group3['Smoking'] = 0      # No smoking
group3['Alcohol'] = 0      # No alcohol
group3['Exercise'] = 1     # Exercises
# Since medical history is already zero, we manually add the bonus of +10 years to Final_Life_Expectancy.
group3['Final_Life_Expectancy'] = np.clip(group3['Final_Life_Expectancy'] + 10, None, 100)
group3.to_csv("synthetic_group3.csv", index=False)

# Merge all groups (total 150 + 250 + 100 = 500 records)
merged_df = pd.concat([group1, group2, group3], ignore_index=True)
merged_df.to_csv("synthetic_merged.csv", index=False)
