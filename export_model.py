import joblib
import numpy as np

model = joblib.load("intruder_model.pkl")

w = model.coef_[0]
b = model.intercept_[0]

np.savetxt("weights.txt", w)
with open("bias.txt","w") as f:
    f.write(str(b))

print("Exported weights and bias")
print("Features:", len(w))
