import pandas as pd
import numpy as np

np.random.seed(42)
n = 500

data = pd.DataFrame({
    "CGPA": np.random.uniform(5.5, 9.8, n),
    "SkillsScore": np.random.randint(40, 100, n),
    "Internships": np.random.randint(0, 4, n),
    "Projects": np.random.randint(1, 6, n),
    "AptitudeScore": np.random.randint(35, 100, n)
})

data["Placed"] = (
    (data["CGPA"] > 7.0) &
    (data["SkillsScore"] > 60) &
    (data["AptitudeScore"] > 55)
).astype(int)

data.to_csv("data/placement_data.csv", index=False)

print("Dataset created!")
