import pandas as pd

def high_HRSD(threshold = 30, gender = 'both', labels_path = 'labels/20251105_d02_questionnaires_app.xlsx'):
    # Read the Excel file
    df = pd.read_excel(labels_path)

    # Initialize lists
    Healthy = []
    Depressed = []


    # Loop through the DataFrame and categorize IDs
    for _, row in df.iterrows():
        if row['HRSD_24.1'] <= 8:
            Healthy.append(row['id'])
        elif row['HRSD_24.1'] > threshold:
            Depressed.append(row['id'])
        else:
            pass

    Healthy = [int(x) for x in Healthy]
    nHealthy = []
    Depressed  = [int(x) for x in Depressed]
    nDepressed = []

    if gender == 'm':
        gender_number = 1
    elif gender == 'w':
        gender_number = 2


    if gender == 'both':
        pass
    else:
        Full_gender = df[df['gender'] == gender_number]['id'].values
        Healthy_new = [x for x in Healthy if x in Full_gender]
        Depressed_new = [x for x in Depressed if x in Full_gender]
        Healthy = Healthy_new
        Depressed = Depressed_new


    for i in Healthy:
        j = "00" + str(i)
        if len(j) == 6:
            k = j[-4:]
        else:
            k = j[-3:]

        nHealthy.append(k)

    for i in Depressed:
        j = "00" + str(i)
        if len(j) == 6:
            k = j[-4:]
        else:
            k = j[-3:]
        nDepressed.append(k)

    Healthy = nHealthy
    Depressed = nDepressed

    return Healthy, Depressed


def detect_Depression( gender = 'both', labels_path = 'labels/20251105_d02_questionnaires_app.xlsx'):
    df = pd.read_excel(labels_path)

    xls_d = df[df['diag']=='d']
    xls_h = df[df['diag']=='nd']

    if gender == 'm':
        gender_number = 1
    elif gender == 'w':
        gender_number = 2

    if gender == 'both':
        Depressed = xls_d['id'].values
        Healthy = xls_h['id'].values
    else:
        Depressed = xls_d[xls_d['gender'] == gender_number]['id'].values
        Healthy = xls_h[xls_h['gender'] == gender_number]['id'].values


    Healthy = [int(x) for x in Healthy]
    nHealthy = []
    Depressed  = [int(x) for x in Depressed]
    nDepressed = []

    for i in Healthy:
        j = "00" + str(i)
        if len(j) == 6:
            k = j[-4:]
        else:
            k = j[-3:]

        nHealthy.append(k)

    for i in Depressed:
        j = "00" + str(i)
        if len(j) == 6:
            k = j[-4:]
        else:
            k = j[-3:]
        nDepressed.append(k)

    Healthy = nHealthy
    Depressed = nDepressed

    return Healthy, Depressed


def detect_symptoms(labels_path = 'labels/20251105_d02_questionnaires_app.xlsx', symptom_name = 'retardation', gender = 'both'):
    # Read the Excel file
    df = pd.read_excel(labels_path)

    # Initialize lists
    Healthy = []
    Depressed = []

    if symptom_name == 'retardation':
        symptom_column = 'D_HRSD_08'
    elif symptom_name == 'insomnia':
        symptom_column = 'D_HRSD_05'
    elif symptom_name == 'agitation':
        symptom_column = 'D_HRSD_09'    
    elif symptom_name == 'weight_loss':
        symptom_column = 'D_HRSD_10'
    else:
        raise ValueError("Invalid symptom name. Choose from 'retardation', 'insomnia', 'agitation', or 'weight_loss'.")
    
    # Loop through the DataFrame and categorize IDs
    for _, row in df.iterrows():
        if row[symptom_column] == 0:
            Healthy.append(row['id'])
        elif row[symptom_column] in [1, 2, 3, 4, 5]:
            Depressed.append(row['id'])
        else:
            pass

    Healthy = [int(x) for x in Healthy]
    nHealthy = []
    Depressed  = [int(x) for x in Depressed]
    nDepressed = []

    if gender == 'm':
        gender_number = 1
    elif gender == 'w':
        gender_number = 2


    if gender == 'both':
        pass
    else:
        Full_gender = df[df['gender'] == gender_number]['id'].values
        Healthy_new = [x for x in Healthy if x in Full_gender]
        Depressed_new = [x for x in Depressed if x in Full_gender]
        Healthy = Healthy_new
        Depressed = Depressed_new

    for i in Healthy:
        j = "00" + str(i)
        if len(j) == 6:
            k = j[-4:]
        else:
            k = j[-3:]

        nHealthy.append(k)

    for i in Depressed:
        j = "00" + str(i)
        if len(j) == 6:
            k = j[-4:]
        else:
            k = j[-3:]
        nDepressed.append(k)

    Healthy = nHealthy
    Depressed = nDepressed

    return Healthy, Depressed


Healthy, Depressed = detect_symptoms(
    symptom_name="retardation",
    gender="both"
)

df = pd.DataFrame({
    "subject_id": Healthy + Depressed,
    "label": ["Healthy"] * len(Healthy) + ["Depressed"] * len(Depressed)
})

df.to_csv("labels/retardation_labels.csv", index=False)

Healthy, Depressed = detect_symptoms(
    symptom_name="agitation",
    gender="both"
)

df = pd.DataFrame({
    "subject_id": Healthy + Depressed,
    "label": ["Healthy"] * len(Healthy) + ["Depressed"] * len(Depressed)
})

df.to_csv("labels/agitation_labels.csv", index=False)