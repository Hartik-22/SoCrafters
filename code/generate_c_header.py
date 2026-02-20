import numpy as np

w = np.loadtxt("weights.txt")
b = float(open("bias.txt").read())

with open("model_params.h","w") as f:
    f.write("#ifndef MODEL_PARAMS_H\n#define MODEL_PARAMS_H\n\n")

    f.write(f"#define FEATURE_SIZE {len(w)}\n\n")

    f.write("float weights[FEATURE_SIZE] = {\n")
    for i,val in enumerate(w):
        f.write(f"{val}f")
        if i != len(w)-1:
            f.write(",")
        if i%8==0:
            f.write("\n")
    f.write("\n};\n\n")

    f.write(f"float bias = {b}f;\n")

    f.write("\n#endif\n")

print("model_params.h generated")
