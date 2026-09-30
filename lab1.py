import numpy as np

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([0, 1, 1, 1])

# Add bias term
Xa = np.hstack([np.ones((4, 1)), X])

# Initial weights: [bias, w1, w2]
w = np.array([-0.5, -1.0, 1.0])

eta = 1.0


def angle(w, x):
    cos_a = (w @ x) / (
        np.linalg.norm(w) * np.linalg.norm(x)
    )
    return np.degrees(
        np.arccos(np.clip(cos_a, -1.0, 1.0))
    )


for epoch in range(1, 21):
    mistakes = 0

    for x, t in zip(Xa, y):
        z = w @ x

        # Step activation
        y_hat = 1 if z >= 0 else 0

        # Error
        e = t - y_hat

        if e != 0:
            a_old = angle(w, x)

            # Perceptron learning rule
            w = w + eta * e * x

            print(
                f"epoch {epoch}: x={x[1:]}, y={t}, "
                f"angle {a_old:.1f} -> {angle(w, x):.1f}, "
                f"w~={w}"
            )

            mistakes += 1

    if mistakes == 0:
        print(f"Converged after {epoch} epochs, w~ = {w}")
        break

else:
    print("No convergence in 20 epochs: data not linearly separable?")


#sample code output:
"""epoch 1: x=[1. 0.], y=1, angle 135.0 -> 71.6, w~=[0.5 0.  1. ]
epoch 2: x=[0. 0.], y=0, angle 63.4 -> 116.6, w~=[-0.5  0.   1. ]
epoch 2: x=[1. 0.], y=1, angle 108.4 -> 45.0, w~=[0.5 1.  1. ]
epoch 3: x=[0. 0.], y=0, angle 70.5 -> 109.5, w~=[-0.5  1.   1. ]
Converged after 4 epochs, w~ = [-0.5  1.   1. ]"""