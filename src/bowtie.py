import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    tsamp = 16 * 16 * 6 * 5.12e-6
    df = 0.1953125 / 4

    f = np.arange(648) * df + 119.48

    kdm = 1 / 2.41e-4
    ddm = 0.1
    
    t = np.arange(-50, 51) * tsamp
    dm = np.arange(-10, 11) * ddm
    
    tm, fm = np.meshgrid(t, f)

    w = 0.01

    print(tm.shape, fm.shape, dm.shape)
    
    fmax = np.max(f)
    tdm = dm[np.newaxis, np.newaxis, :] * kdm * (fm[:, :, np.newaxis]**(-2) - fmax**(-2))

    z = np.mean(np.exp(-0.5 * ((tm[:, :, np.newaxis] - tdm) / w)**2), axis=0)
    
    fig, ax = plt.subplots()

    ax.imshow(z.T, origin="lower", aspect="auto")

    plt.savefig('output.png')  # Save to an image file

    